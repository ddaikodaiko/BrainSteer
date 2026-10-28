"""
latentflow.core.steering
========================

Núcleo matemático de BrainSteer: steering de queries con proyección ortogonal
al subespacio de diferencias de keys (SKOP).

Pipeline
--------
1. Calibración offline (``SKOPEngine.calibrate``)
   Para cada par contrastivo (prompt positivo, prompt negativo) se capturan las
   salidas de ``q_proj`` y ``k_proj`` de cada capa y, por cabeza, se acumula en
   CPU/float64::

       Sigma_ΔK = E[Δk Δk^T],      Δk = k_pos - k_neg     (por cabeza KV)
       r_q      = E[q_pos - q_neg]                        (por cabeza de query)

2. Proyector ortogonal (``compute_orthogonal_projector``)
   ``eigh(Sigma_ΔK) = V Λ V^T``; se retienen las k primeras direcciones cuya
   energía acumulada alcanza ``gamma_energy`` y ``P_ΔK = I - U_k U_k^T``.

3. Filtrado por cociente de Rayleigh (``SKOPEngine.compute_risk``)
   ``R_h = r_h^T Σ r_h / r_h^T r_h``. Las cabezas por encima del percentil
   ``risk_percentile`` (top 20 % por defecto, global sobre todas las capas)
   reciben ``q' = q + α P_ΔK r_q``; el resto, steering aditivo ``q' = q + α r_q``.

Como ``P_ΔK`` y ``r_q`` son constantes, el vector efectivo de cada cabeza se
precalcula en float64 y se castea una única vez al dtype del modelo
(float16/bfloat16). En inferencia el coste es una suma vectorial por capa:
ninguna matriz d×d llega a la VRAM.

Nota: los hooks actúan sobre las salidas de ``q_proj``/``k_proj``, es decir,
en el espacio previo a RoPE.

Uso
---
>>> model, tok = load_quantized_model("google/gemma-2-2b-it")
>>> engine = SKOPEngine(model, SKOPConfig(alpha=4.0), tokenizer=tok)
>>> engine.calibrate(positive_prompts, negative_prompts).fit()
>>> with engine.steering():
...     out = model.generate(**tok("Hola", return_tensors="pt").to(model.device))
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

__all__ = [
    "DEFAULT_MODEL_ID",
    "AttentionGeometry",
    "ProjectorResult",
    "SecondMomentAccumulator",
    "SKOPConfig",
    "SKOPEngine",
    "compute_orthogonal_projector",
    "load_quantized_model",
    "rayleigh_quotient",
    "select_risky_heads",
    "trace_normalize",
]

DEFAULT_MODEL_ID = "google/gemma-2-2b-it"

TokenAggregation = Literal["last", "mean"]

# Toda la álgebra lineal se hace en CPU con esta precisión.
_MATH_DTYPE = torch.float64

# Rutas habituales a la lista de bloques decoder en modelos HF.
_DECODER_LAYER_PATHS = (
    "model.layers",                 # Llama, Gemma, Gemma-2, Mistral, Qwen2
    "model.model.layers",           # wrappers PEFT
    "model.language_model.layers",  # Gemma-3 / multimodales
    "language_model.model.layers",
    "layers",
)


# --------------------------------------------------------------------------- #
# Configuración y geometría
# --------------------------------------------------------------------------- #
@dataclass
class SKOPConfig:
    """Hiperparámetros de SKOP.

    Attributes:
        gamma_energy: fracción de energía espectral de Sigma_ΔK que se elimina.
        risk_percentile: percentil del cociente de Rayleigh a partir del cual una
            cabeza se considera de riesgo (0.80 -> top 20 %).
        alpha: intensidad del steering.
        layers: índices de capa a intervenir (``None`` = todas).
        token_aggregation: cómo resumir cada secuencia en un vector
            (``"last"``: último token no-pad; ``"mean"``: media enmascarada).
        batch_size: pares por forward durante la calibración.
        max_length: truncado de tokenización en calibración.
        normalize_steering: si True, ``r_q`` se normaliza a norma 1 por cabeza.
        normalize_rayleigh: si True, el cociente de Rayleigh se evalúa sobre
            ``Σ / (tr(Σ) + rayleigh_eps)`` para que el ranking global de riesgo no
            se sesgue hacia las capas con mayor escala de activación.
        rayleigh_eps: estabilizador de la normalización por traza.
        eps: estabilizador numérico.
    """

    gamma_energy: float = 0.90
    risk_percentile: float = 0.80
    alpha: float = 1.0
    layers: Optional[Sequence[int]] = None
    token_aggregation: TokenAggregation = "last"
    batch_size: int = 4
    max_length: int = 256
    normalize_steering: bool = False
    normalize_rayleigh: bool = True
    rayleigh_eps: float = 1e-6
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma_energy <= 1.0:
            raise ValueError(f"gamma_energy debe estar en (0, 1], recibido {self.gamma_energy}")
        if not 0.0 <= self.risk_percentile <= 1.0:
            raise ValueError(f"risk_percentile debe estar en [0, 1], recibido {self.risk_percentile}")
        if self.token_aggregation not in ("last", "mean"):
            raise ValueError(f"token_aggregation desconocido: {self.token_aggregation!r}")
        if self.batch_size < 1:
            raise ValueError("batch_size debe ser >= 1")


@dataclass(frozen=True)
class AttentionGeometry:
    """Dimensiones de la atención (soporta GQA: varias queries por cabeza KV)."""

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads debe ser múltiplo de num_kv_heads")

    @property
    def group_size(self) -> int:
        return self.num_heads // self.num_kv_heads

    def kv_index(self) -> Tensor:
        """Cabeza KV que usa cada cabeza de query, shape (num_heads,)."""
        return torch.arange(self.num_heads) // self.group_size

    @classmethod
    def from_config(cls, config: Any) -> "AttentionGeometry":
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", None) or num_heads
        # Gemma-2 declara head_dim explícito (256 != hidden_size / num_heads).
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // num_heads
        return cls(config.num_hidden_layers, num_heads, num_kv_heads, head_dim)


# --------------------------------------------------------------------------- #
# Acumuladores en CPU
# --------------------------------------------------------------------------- #
class SecondMomentAccumulator:
    """Acumula ``sum(Δ Δ^T)`` por cabeza en CPU/float64; nunca reserva VRAM."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        self.sum = torch.zeros(num_heads, head_dim, head_dim, dtype=_MATH_DTYPE)
        self.count = 0

    @torch.no_grad()
    def update(self, delta: Tensor) -> None:
        """``delta``: (N, H, d), en cualquier device/dtype."""
        delta = delta.detach().to(device="cpu", dtype=_MATH_DTYPE)
        self.sum.add_(torch.einsum("nhd,nhe->hde", delta, delta))
        self.count += delta.shape[0]

    def finalize(self) -> Tensor:
        """Segundo momento ``E[Δ Δ^T]``, shape (H, d, d), simétrico."""
        if self.count == 0:
            raise RuntimeError("No se ha acumulado ninguna muestra")
        sigma = self.sum / self.count
        return 0.5 * (sigma + sigma.mT)


class _MeanAccumulator:
    def __init__(self, num_heads: int, head_dim: int) -> None:
        self.sum = torch.zeros(num_heads, head_dim, dtype=_MATH_DTYPE)
        self.count = 0

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        x = x.detach().to(device="cpu", dtype=_MATH_DTYPE)
        self.sum.add_(x.sum(0))
        self.count += x.shape[0]

    def finalize(self) -> Tensor:
        if self.count == 0:
            raise RuntimeError("No se ha acumulado ninguna muestra")
        return self.sum / self.count


# --------------------------------------------------------------------------- #
# Núcleo matemático (funciones puras, batched sobre cabezas)
# --------------------------------------------------------------------------- #
@dataclass
class ProjectorResult:
    """Resultado de la descomposición espectral de Sigma_ΔK.

    Attributes:
        projector: ``P = I - U_k U_k^T``, shape (..., d, d), float64.
        eigenvalues: autovalores en orden descendente, shape (..., d).
        eigenvectors: autovectores (columnas) en el mismo orden, shape (..., d, d).
        rank: número de direcciones retenidas k, shape (...).
    """

    projector: Tensor
    eigenvalues: Tensor
    eigenvectors: Tensor
    rank: Tensor

    def retained_basis(self) -> Tensor:
        """``U_k`` con las columnas no retenidas a cero, shape (..., d, d)."""
        d = self.eigenvalues.shape[-1]
        mask = torch.arange(d) < self.rank.unsqueeze(-1)
        return self.eigenvectors * mask.unsqueeze(-2).to(self.eigenvectors.dtype)


@torch.no_grad()
def compute_orthogonal_projector(
    sigma: Tensor, gamma_energy: float = 0.90, eps: float = 1e-12
) -> ProjectorResult:
    """Proyector sobre el complemento ortogonal del subespacio dominante de Σ.

    Args:
        sigma: matriz(es) simétricas PSD, shape (..., d, d). Se suben a float64/CPU
            aunque lleguen en float16/bfloat16.
        gamma_energy: energía acumulada a retener en ``U_k``.
        eps: si la energía total es <= eps no se retiene nada (P = I).
    """
    if not 0.0 < gamma_energy <= 1.0:
        raise ValueError(f"gamma_energy debe estar en (0, 1], recibido {gamma_energy}")
    s = sigma.detach().to(device="cpu", dtype=_MATH_DTYPE)
    s = 0.5 * (s + s.mT)
    d = s.shape[-1]

    eigenvalues, eigenvectors = torch.linalg.eigh(s)  # orden ascendente
    eigenvalues = eigenvalues.flip(-1).clamp_min(0.0)  # PSD: descarta ruido negativo
    eigenvectors = eigenvectors.flip(-1)

    total = eigenvalues.sum(-1, keepdim=True)
    cumulative = eigenvalues.cumsum(-1) / total.clamp_min(eps)
    # k = mínimo número de componentes con energía acumulada >= gamma.
    rank = ((cumulative < gamma_energy - 1e-12).sum(-1) + 1).clamp_max(d)
    rank = torch.where(total.squeeze(-1) > eps, rank, torch.zeros_like(rank))

    mask = (torch.arange(d) < rank.unsqueeze(-1)).to(_MATH_DTYPE)
    basis = eigenvectors * mask.unsqueeze(-2)
    eye = torch.eye(d, dtype=_MATH_DTYPE).expand_as(s)
    projector = eye - basis @ basis.mT
    # Simetría exacta bit a bit (la suma en coma flotante es conmutativa).
    projector = 0.5 * (projector + projector.mT)
    return ProjectorResult(projector, eigenvalues, eigenvectors, rank)


@torch.no_grad()
def rayleigh_quotient(sigma: Tensor, r: Tensor, eps: float = 1e-12) -> Tensor:
    """``R(r) = r^T Σ r / r^T r``. sigma (..., d, d), r (..., d) -> (...)."""
    s = sigma.detach().to(device="cpu", dtype=_MATH_DTYPE)
    v = r.detach().to(device="cpu", dtype=_MATH_DTYPE)
    numerator = torch.einsum("...i,...ij,...j->...", v, s, v)
    return numerator / (v * v).sum(-1).clamp_min(eps)


@torch.no_grad()
def trace_normalize(sigma: Tensor, eps: float = 1e-6) -> Tensor:
    """``Σ / (tr(Σ) + eps)`` por matriz, shape (..., d, d), float64/CPU.

    El cociente de Rayleigh sobre la matriz normalizada vale ``R / tr(Σ)`` y queda
    en ``[0, λ_max / tr(Σ)] ⊆ [0, 1]``: la fracción de la energía de ΔK que el
    vector captura, comparable entre capas de escalas distintas.
    """
    s = sigma.detach().to(device="cpu", dtype=_MATH_DTYPE)
    trace = s.diagonal(dim1=-2, dim2=-1).sum(-1)
    return s / (trace + eps)[..., None, None]


@torch.no_grad()
def select_risky_heads(scores: Tensor, percentile: float = 0.80) -> Tuple[Tensor, float]:
    """Máscara booleana de cabezas cuyo score supera el percentil dado."""
    flat = scores.detach().to(device="cpu", dtype=_MATH_DTYPE).flatten()
    threshold = torch.quantile(flat, percentile).item()
    return scores.cpu() > threshold, threshold


def _aggregate_tokens(
    output: Tensor, attention_mask: Tensor, num_heads: int, head_dim: int, mode: TokenAggregation
) -> Tensor:
    """(B, T, H*d) -> (B, H, d) en float32, respetando padding izq. o der."""
    batch, seq_len, _ = output.shape
    x = output.reshape(batch, seq_len, num_heads, head_dim)
    mask = attention_mask.to(x.device)
    if mode == "last":
        positions = torch.arange(1, seq_len + 1, device=x.device)
        last = (mask.long() * positions).argmax(-1)
        return x[torch.arange(batch, device=x.device), last].float()
    weights = mask.float()[:, :, None, None]
    return (x.float() * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class _StopForward(Exception):
    """Corta el forward tras la última capa calibrada (ahorra cómputo y logits)."""


def _raise_stop(module: nn.Module, args: Any, output: Any) -> None:
    raise _StopForward


def _resolve_decoder_layers(model: nn.Module, layers_path: Optional[str]) -> nn.ModuleList:
    for path in (layers_path,) if layers_path else _DECODER_LAYER_PATHS:
        obj: Any = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and len(obj) > 0 and hasattr(obj[0], "self_attn"):
            return obj
    raise ValueError("No se encontraron bloques decoder con `self_attn`; indica `layers_path`")


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #
class SKOPEngine:
    """Calibra, ajusta y aplica steering SKOP sobre las queries de un LLM HF.

    Funciona con modelos cuantizados (bitsandbytes 4/8 bit) porque sólo engancha
    forward hooks a las salidas de ``q_proj``/``k_proj``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[SKOPConfig] = None,
        tokenizer: Any = None,
        layers_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SKOPConfig()
        self.geometry = AttentionGeometry.from_config(model.config)
        self.decoder_layers = _resolve_decoder_layers(model, layers_path)

        n_layers = len(self.decoder_layers)
        layers = range(n_layers) if self.config.layers is None else self.config.layers
        self.layer_ids: List[int] = sorted({int(i) % n_layers for i in layers})
        if not self.layer_ids:
            raise ValueError("No hay capas que intervenir")

        self.sigma_delta_k: Dict[int, Tensor] = {}      # (H_kv, d, d)
        self.steering_vectors: Dict[int, Tensor] = {}   # r_q, (H, d)
        self.projectors: Dict[int, ProjectorResult] = {}
        self.rayleigh: Dict[int, Tensor] = {}           # (H,)
        self.risk_mask: Dict[int, Tensor] = {}          # (H,) bool
        self.risk_threshold: Optional[float] = None
        self.num_calibration_pairs = 0

        self._sigma_acc: Dict[int, SecondMomentAccumulator] = {}
        self._rq_acc: Dict[int, _MeanAccumulator] = {}
        self._effective: Dict[int, Tensor] = {}         # (H*d,) float64, sin α
        self._risk_sigma: Dict[int, Tensor] = {}        # Σ usada en Rayleigh, por cabeza de query (H, d, d)
        self._cast_cache: Dict[Tuple[int, torch.device, torch.dtype], Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._telemetry_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._probe: Dict[int, Tensor] = {}             # última q (post-steering) por capa, en device
        self._alpha = float(self.config.alpha)
        self._use_projection = True
        self._layer_plan: Optional[Dict[int, Tuple[float, bool]]] = None  # capa -> (alpha, proyección)
        self.reset_calibration()

    # ----------------------------------------------------------------- alpha
    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = float(value)
        self._cast_cache.clear()

    @property
    def use_projection(self) -> bool:
        """Filtro SKOP activo: las cabezas de riesgo reciben ``P r_q`` en vez de ``r_q``."""
        return self._use_projection

    @use_projection.setter
    def use_projection(self, value: bool) -> None:
        self._use_projection = bool(value)
        if self._effective:
            self._build_effective_vectors()

    # ------------------------------------------------------- plan por capa
    @property
    def layer_plan(self) -> Optional[Dict[int, Tuple[float, bool]]]:
        return None if self._layer_plan is None else dict(self._layer_plan)

    @property
    def active_layers(self) -> List[int]:
        """Capas que reciben steering: todas las calibradas o sólo las del plan."""
        return list(self.layer_ids) if self._layer_plan is None else sorted(self._layer_plan)

    def alpha_for(self, layer: int) -> float:
        if self._layer_plan is not None and layer in self._layer_plan:
            return self._layer_plan[layer][0]
        return self._alpha

    def projection_for(self, layer: int) -> bool:
        if self._layer_plan is not None and layer in self._layer_plan:
            return self._layer_plan[layer][1]
        return self._use_projection

    def set_layer_plan(self, plan: Optional[Mapping[int, Tuple[float, bool]]]) -> "SKOPEngine":
        """Alpha y modo (proyección SKOP o aditivo) por capa.

        Con un plan, sólo sus capas reciben steering y ``alpha``/``use_projection``
        globales se ignoran. ``None`` vuelve al modo global (todas las capas calibradas).
        Si los hooks están activos se re-registran sobre las nuevas capas.
        """
        if plan is not None:
            unknown = sorted(set(plan) - set(self.layer_ids))
            if unknown:
                raise KeyError(f"Capas sin calibrar en el plan: {unknown}")
            plan = {int(l): (float(a), bool(p)) for l, (a, p) in plan.items()}
        self._layer_plan = plan
        if self._effective:
            self._build_effective_vectors()
        self._cast_cache.clear()
        if self.is_attached:
            self.attach()
        return self

    @property
    def is_attached(self) -> bool:
        return bool(self._handles)

    @property
    def is_fitted(self) -> bool:
        return bool(self._effective)

    @property
    def telemetry_attached(self) -> bool:
        return bool(self._telemetry_handles)

    # ----------------------------------------------------------- calibración
    def reset_calibration(self) -> None:
        g = self.geometry
        self._sigma_acc = {l: SecondMomentAccumulator(g.num_kv_heads, g.head_dim) for l in self.layer_ids}
        self._rq_acc = {l: _MeanAccumulator(g.num_heads, g.head_dim) for l in self.layer_ids}

    @torch.no_grad()
    def calibrate(
        self,
        positive_prompts: Sequence[str],
        negative_prompts: Sequence[str],
        reset: bool = True,
    ) -> "SKOPEngine":
        """Calibración offline completa a partir de prompts contrastivos alineados."""
        if self.tokenizer is None:
            raise RuntimeError("calibrate() necesita un tokenizer; usa accumulate_pair_batch()")
        if len(positive_prompts) != len(negative_prompts):
            raise ValueError("positive_prompts y negative_prompts deben tener la misma longitud")
        if getattr(self.tokenizer, "pad_token", "") is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Llama-3 no trae pad
        if reset:
            self.reset_calibration()

        bs = self.config.batch_size
        for start in range(0, len(positive_prompts), bs):
            self.accumulate_pair_batch(
                self._encode(positive_prompts[start : start + bs]),
                self._encode(negative_prompts[start : start + bs]),
            )
        return self.finalize_calibration()

    def _encode(self, prompts: Sequence[str]) -> Dict[str, Tensor]:
        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    @torch.no_grad()
    def accumulate_pair_batch(
        self, positive_inputs: Mapping[str, Tensor], negative_inputs: Mapping[str, Tensor]
    ) -> None:
        """Acumula un lote de pares ya tokenizados (``input_ids``/``attention_mask``)."""
        q_pos, k_pos = self._capture(positive_inputs)
        q_neg, k_neg = self._capture(negative_inputs)
        for l in self.layer_ids:
            self._sigma_acc[l].update(k_pos[l].to(_MATH_DTYPE) - k_neg[l].to(_MATH_DTYPE))
            self._rq_acc[l].update(q_pos[l].to(_MATH_DTYPE) - q_neg[l].to(_MATH_DTYPE))

    def _capture(self, inputs: Mapping[str, Tensor]) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
        """Un forward; devuelve q y k agregados por secuencia, ya en CPU (float32)."""
        if self.is_attached:
            raise RuntimeError("Llama a detach() antes de calibrar: el steering contaminaría Σ")
        device = self._input_device()
        input_ids = inputs["input_ids"].to(device)
        mask = inputs.get("attention_mask")
        mask = torch.ones_like(input_ids) if mask is None else mask.to(device)

        g, mode = self.geometry, self.config.token_aggregation
        captured: Dict[str, Dict[int, Tensor]] = {"q": {}, "k": {}}

        def make_hook(kind: str, layer: int, n_heads: int):
            def hook(module: nn.Module, args: Any, output: Tensor) -> None:
                captured[kind][layer] = _aggregate_tokens(output, mask, n_heads, g.head_dim, mode).cpu()

            return hook

        handles = []
        try:
            for l in self.layer_ids:
                attn = self.decoder_layers[l].self_attn
                handles.append(attn.q_proj.register_forward_hook(make_hook("q", l, g.num_heads)))
                handles.append(attn.k_proj.register_forward_hook(make_hook("k", l, g.num_kv_heads)))
            handles.append(self.decoder_layers[self.layer_ids[-1]].register_forward_hook(_raise_stop))
            try:
                self.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
            except _StopForward:
                pass
        finally:
            for h in handles:
                h.remove()

        missing = [l for l in self.layer_ids if l not in captured["q"] or l not in captured["k"]]
        if missing:
            raise RuntimeError(f"No se capturaron activaciones en las capas {missing}")
        return captured["q"], captured["k"]

    def _input_device(self) -> torch.device:
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings().weight.device
        return next(self.model.parameters()).device

    def finalize_calibration(self) -> "SKOPEngine":
        """Convierte los acumuladores en Sigma_ΔK y r_q."""
        for l in self.layer_ids:
            self.sigma_delta_k[l] = self._sigma_acc[l].finalize()
            r = self._rq_acc[l].finalize()
            if self.config.normalize_steering:
                r = r / r.norm(dim=-1, keepdim=True).clamp_min(self.config.eps)
            self.steering_vectors[l] = r
        self.num_calibration_pairs = self._sigma_acc[self.layer_ids[0]].count
        return self

    def set_steering_vectors(self, vectors: Mapping[int, Tensor]) -> "SKOPEngine":
        """Sustituye ``r_q`` por vectores externos, shape (H, d) o (H*d,) por capa."""
        g = self.geometry
        for l, v in vectors.items():
            if l not in self.layer_ids:
                raise KeyError(f"La capa {l} no está en layer_ids")
            self.steering_vectors[l] = v.detach().to("cpu", _MATH_DTYPE).reshape(g.num_heads, g.head_dim)
        if self.projectors:
            self.compute_risk()
        return self

    # ------------------------------------------------------------ ajuste SKOP
    def fit(self) -> "SKOPEngine":
        """Proyectores + filtrado de Rayleigh + vectores efectivos."""
        return self.fit_projectors().compute_risk()

    def fit_projectors(self) -> "SKOPEngine":
        if not self.sigma_delta_k:
            raise RuntimeError("Ejecuta calibrate() antes de fit()")
        for l in self.layer_ids:
            self.projectors[l] = compute_orthogonal_projector(
                self.sigma_delta_k[l], self.config.gamma_energy, self.config.eps
            )
        return self

    def compute_risk(self) -> "SKOPEngine":
        """Cociente de Rayleigh por cabeza de query y máscara global top-(1-p)."""
        if not self.projectors or not self.steering_vectors:
            raise RuntimeError("Faltan proyectores o vectores de steering")
        kv_idx = self.geometry.kv_index()
        for l in self.layer_ids:
            sigma = self.sigma_delta_k[l]
            if self.config.normalize_rayleigh:
                sigma = trace_normalize(sigma, self.config.rayleigh_eps)
            # GQA: cada cabeza de query se evalúa con la Σ de su grupo KV.
            self._risk_sigma[l] = sigma[kv_idx]
            self.rayleigh[l] = rayleigh_quotient(self._risk_sigma[l], self.steering_vectors[l], self.config.eps)
        scores = torch.stack([self.rayleigh[l] for l in self.layer_ids])  # (L, H)
        mask, self.risk_threshold = select_risky_heads(scores, self.config.risk_percentile)
        for i, l in enumerate(self.layer_ids):
            self.risk_mask[l] = mask[i]
        self._build_effective_vectors()
        return self

    def _build_effective_vectors(self) -> None:
        kv_idx = self.geometry.kv_index()
        for l in self.layer_ids:
            r = self.steering_vectors[l]
            projected = torch.einsum("hij,hj->hi", self.projectors[l].projector[kv_idx], r)
            use_projection = self.risk_mask[l] & self.projection_for(l)
            effective = torch.where(use_projection.unsqueeze(-1), projected, r)
            self._effective[l] = effective.reshape(-1).contiguous()
        self._cast_cache.clear()

    def effective_vectors(self) -> Dict[int, Tensor]:
        """``α · v_h`` por capa, shape (H, d), float64 (útil para inspección)."""
        g = self.geometry
        return {l: self.alpha_for(l) * v.reshape(g.num_heads, g.head_dim) for l, v in self._effective.items()}

    # --------------------------------------------------------------- steering
    def attach(self, alpha: Optional[float] = None) -> "SKOPEngine":
        """Registra los hooks de steering en ``q_proj``."""
        if not self._effective:
            raise RuntimeError("Ejecuta fit() antes de attach()")
        if alpha is not None:
            self.alpha = alpha
        self.detach()
        for l in self.active_layers:
            q_proj = self.decoder_layers[l].self_attn.q_proj
            # prepend: el steering corre antes que cualquier otro hook (p. ej. la telemetría).
            self._handles.append(q_proj.register_forward_hook(self._make_steer_hook(l), prepend=True))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextmanager
    def steering(self, alpha: Optional[float] = None) -> Iterator["SKOPEngine"]:
        self.attach(alpha)
        try:
            yield self
        finally:
            self.detach()

    def _make_steer_hook(self, layer: int):
        def hook(module: nn.Module, args: Any, output: Tensor) -> Tensor:
            return output + self._steering_tensor(layer, output.device, output.dtype)

        return hook

    def _steering_tensor(self, layer: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (layer, device, dtype)
        cached = self._cast_cache.get(key)
        if cached is None:
            # α se aplica en float64 y se castea una sola vez al dtype del modelo.
            cached = (self.alpha_for(layer) * self._effective[layer]).to(device=device, dtype=dtype)
            if not torch.isfinite(cached).all():
                raise ValueError(f"El vector de steering de la capa {layer} desborda {dtype}; reduce alpha")
            self._cast_cache[key] = cached
        return cached

    # ------------------------------------------------------------ telemetría
    def attach_telemetry(self) -> "SKOPEngine":
        """Sondas en ``q_proj`` que guardan la q del último token (ya steerada).

        La copia se queda en el device del modelo, sin sincronizar;
        ``collect_telemetry()`` la baja a CPU una vez por paso de generación.
        """
        if not self._risk_sigma:
            raise RuntimeError("Ejecuta fit() antes de attach_telemetry()")
        self.detach_telemetry()
        for l in self.layer_ids:
            q_proj = self.decoder_layers[l].self_attn.q_proj
            self._telemetry_handles.append(q_proj.register_forward_hook(self._make_probe_hook(l)))
        return self

    def detach_telemetry(self) -> None:
        for h in self._telemetry_handles:
            h.remove()
        self._telemetry_handles.clear()
        self._probe.clear()

    def _make_probe_hook(self, layer: int):
        def hook(module: nn.Module, args: Any, output: Tensor) -> None:
            self._probe[layer] = output[:, -1].detach().clone()

        return hook

    @torch.no_grad()
    def collect_telemetry(self, clear: bool = True) -> Dict[int, Tensor]:
        """Cociente de Rayleigh en vivo de la query actual, por capa: (B, H) float64.

        Usa la misma Σ (normalizada por traza si procede) que el ranking de riesgo,
        así que es directamente comparable con ``rayleigh`` y ``risk_threshold``.
        """
        g = self.geometry
        live: Dict[int, Tensor] = {}
        for l, q in self._probe.items():
            qh = q.to(device="cpu", dtype=_MATH_DTYPE).reshape(q.shape[0], g.num_heads, g.head_dim)
            sigma = self._risk_sigma[l].expand(qh.shape[0], -1, -1, -1)
            live[l] = rayleigh_quotient(sigma, qh, self.config.eps)
        if clear:
            self._probe.clear()
        return live

    # ------------------------------------------------------------ inspección
    def summary(self) -> Dict[str, Any]:
        return {
            "num_calibration_pairs": self.num_calibration_pairs,
            "gamma_energy": self.config.gamma_energy,
            "risk_percentile": self.config.risk_percentile,
            "risk_threshold": self.risk_threshold,
            "normalize_rayleigh": self.config.normalize_rayleigh,
            "use_projection": self._use_projection,
            "alpha": self._alpha,
            "layer_plan": self.layer_plan,
            "layers": {
                l: {
                    "rank_per_kv_head": self.projectors[l].rank.tolist(),
                    "rayleigh": self.rayleigh[l].tolist(),
                    "risky_heads": self.risk_mask[l].nonzero().flatten().tolist(),
                }
                for l in self.layer_ids
                if l in self.projectors and l in self.rayleigh
            },
        }

    # ---------------------------------------------------------- persistencia
    def state_dict(self) -> Dict[str, Any]:
        return {
            "layer_ids": list(self.layer_ids),
            "geometry": vars(self.geometry).copy(),
            "num_calibration_pairs": self.num_calibration_pairs,
            "sigma_delta_k": dict(self.sigma_delta_k),
            "steering_vectors": dict(self.steering_vectors),
        }

    def load_state_dict(self, state: Mapping[str, Any], fit: bool = True) -> "SKOPEngine":
        if state["geometry"] != vars(self.geometry):
            raise ValueError("La calibración pertenece a un modelo con otra geometría de atención")
        if list(state["layer_ids"]) != self.layer_ids:
            raise ValueError("La calibración cubre otras capas")
        self.sigma_delta_k = {int(l): t.to(_MATH_DTYPE) for l, t in state["sigma_delta_k"].items()}
        self.steering_vectors = {int(l): t.to(_MATH_DTYPE) for l, t in state["steering_vectors"].items()}
        self.num_calibration_pairs = state["num_calibration_pairs"]
        return self.fit() if fit else self

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, fit: bool = True) -> "SKOPEngine":
        return self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), fit=fit)


# --------------------------------------------------------------------------- #
# Carga de modelos cuantizados
# --------------------------------------------------------------------------- #
def load_quantized_model(
    model_id: str = DEFAULT_MODEL_ID,
    load_in_4bit: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16,
    device_map: Any = "auto",
    attn_implementation: str = "eager",
):
    """Carga un modelo HF cuantizado con bitsandbytes para una GPU de ~8 GB.

    Gemma-2-2B-IT cabe holgado en 4 bit; Llama-3.1-8B en 4 bit ocupa ~5.5 GB.
    Usa ``compute_dtype=torch.float16`` en GPUs sin bfloat16 (pre-Ampere), pero
    ten en cuenta que Gemma-2 es propensa a desbordar en float16. Gemma-2
    recomienda ``attn_implementation="eager"`` por el soft-capping de logits.
    """
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization = BitsAndBytesConfig(load_in_8bit=True)

    # `torch_dtype` pasó a llamarse `dtype` en transformers 4.56.
    dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map=device_map,
        attn_implementation=attn_implementation,
        **{dtype_kw: compute_dtype},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
