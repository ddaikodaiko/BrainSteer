"""
latentflow.core.steering
========================

Núcleo matemático de BrainSteer: steering de queries con proyección ortogonal
al subespacio de diferencias de keys (SKOP; Luo, Espinosa Zarlenga y Jamnik,
arXiv:2605.06342, Algoritmo 1).

Pipeline
--------
1. Calibración offline con el modelo base, sin steering (``SKOPEngine.calibrate``).

   a) Geometría de keys sobre un *utility set* genérico (``accumulate_utility_batch``).
      Para cada cabeza de atención, cada paso (query) ``t`` y su distribución de
      atención real ``a_t``, el *Focus Set* ``F_t`` es el conjunto mínimo de keys
      que acumula al menos ``tau_high`` (0.8) de la masa, y el *Tail Set* ``T_t``
      el resto de keys atendibles. En CPU/float64 se acumula el segundo momento,
      con cada paso promediado sobre sus pares y los pasos con el mismo peso::

          Sigma_ΔK = E_t E_{(i,j) ~ U(F_t × T_t)}[Δk Δkᵀ],   Δk = k_i − k_j   (por cabeza)

      No intervienen prompts contrastivos: Σ_ΔK describe la geometría con la que el
      modelo base separa los tokens que atiende de los que no.

   b) Dirección conceptual ``r_q``, externa a Σ_ΔK. O bien se estima con pares
      contrastivos de concepto (``accumulate_concept_batch``)::

          r_q = E[q_pos − q_neg]                                          (por cabeza de query)

      o bien se pasa ya calculada a ``fit(steering_vectors=...)``. Una dirección
      del residual stream (p. ej. el decoder ``W_dec[f]`` de un SAE de Gemma Scope)
      se lleva al espacio de queries de cada capa con ``residual_to_query``::

          r_q = q_proj(input_layernorm(v))

      y se inyecta en caliente con ``set_layer_directions``.

2. Proyector ortogonal (``compute_orthogonal_projector``)
   ``eigh(Sigma_ΔK) = V Λ Vᵀ``; se retienen las k primeras direcciones cuya
   energía acumulada alcanza ``gamma_energy`` y ``P_ΔK = I − U_k U_kᵀ``.

3. Filtrado por cociente de Rayleigh (``SKOPEngine.compute_risk``)
   ``R_h = r_hᵀ Σ r_h / (‖r_h‖² + ε)``: la energía absoluta de la perturbación del
   gap Focus/Tail por unidad de norma (``normalize_rayleigh=True`` usa
   ``Σ / tr(Σ)``, opcional). Las cabezas por encima del percentil
   ``risk_percentile`` (top 20 % por defecto) entre las de las capas que reciben
   steering (todas en modo global, como en el paper) reciben
   ``q' = q + α P_ΔK r_q``; el resto, ``q' = q + α r_q``.

Por qué funciona: ``q'·Δk = q·Δk + α (P_ΔK r_q)·Δk``. Como ``P_ΔK`` anula el
subespacio donde se concentra Δk, el steering apenas cambia la diferencia de score
entre tokens Focus y Tail, es decir, preserva el patrón de atención del modelo base.

Como ``P_ΔK`` y ``r_q`` son constantes, el vector efectivo de cada cabeza se
precalcula en float64 y se castea una única vez al dtype del modelo
(float16/bfloat16). En inferencia el coste es una suma vectorial por capa:
ninguna matriz d×d llega a la VRAM.

Notas:
- Las keys y queries se leen de las salidas de ``q_proj``/``k_proj`` (espacio
  previo a RoPE, donde actúa el steering). Las probabilidades de atención que
  definen Focus/Tail son las reales del modelo (post-RoPE, con soft-capping y
  ventana deslizante en Gemma-2), por lo que la calibración de Σ_ΔK requiere
  ``attn_implementation="eager"`` (SDPA/Flash no exponen los pesos).
- RoPE: Σ_ΔK se estima con keys previas a RoPE y el steering se suma a la query
  previa a RoPE. En el score real, la perturbación de la query ``t`` frente a la
  key ``i`` es ``δq · R_{i−t} k_i`` (``R`` = rotación relativa), así que ``P_ΔK``
  protege exactamente sólo donde esa rotación es pequeña (las frecuencias bajas
  de RoPE); en las altas, la protección es aproximada.

Uso
---
>>> model, tok = load_quantized_model("google/gemma-2-2b-it")
>>> engine = SKOPEngine(model, SKOPConfig(alpha=4.0), tokenizer=tok)
>>> engine.calibrate(utility_texts, positive_prompts, negative_prompts).fit()
>>> with engine.steering():
...     out = model.generate(**tok("Hola", return_tensors="pt").to(model.device))
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
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
    "focus_set_mask",
    "focus_tail_second_moment",
    "load_quantized_model",
    "rayleigh_quotient",
    "select_risky_heads",
    "trace_normalize",
]

DEFAULT_MODEL_ID = "google/gemma-2-2b-it"

#: Hiperparámetros del ajuste (proyectores y riesgo) que viajan con el artefacto.
FIT_CONFIG_KEYS = ("gamma_energy", "risk_percentile", "normalize_rayleigh")

#: Formato del artefacto de ``SKOPEngine.save()``. v3: Focus Set por masa acumulada,
#: media por pasos y Σ_ΔK por cabeza de query (v2 no se puede convertir: recalibra).
ARTIFACT_FORMAT = "latentflow.skop/v3"

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
        tau_high: masa de atención del Focus Set: para cada cabeza y paso, el
            conjunto mínimo de keys que acumula al menos esa fracción (0.8 en SKOP).
            El resto de keys atendibles forma el Tail Set.
        token_aggregation: cómo resumir cada prompt de concepto en un vector para
            ``r_q`` (``"last"``: último token no-pad; ``"mean"``: media enmascarada).
        batch_size: secuencias (o pares de concepto) por forward en calibración.
        max_length: truncado de tokenización en calibración.
        normalize_steering: si True, ``r_q`` se normaliza a norma 1 por cabeza.
        normalize_rayleigh: si True (opcional, desactivado como en SKOP), el
            cociente de Rayleigh se evalúa sobre ``Σ / (tr(Σ) + rayleigh_eps)``, de
            modo que el ranking global no favorece a las capas de mayor escala.
        rayleigh_eps: estabilizador de la normalización por traza.
        eps: estabilizador numérico.
    """

    gamma_energy: float = 0.90
    risk_percentile: float = 0.80
    alpha: float = 1.0
    layers: Optional[Sequence[int]] = None
    tau_high: float = 0.8
    token_aggregation: TokenAggregation = "last"
    batch_size: int = 4
    max_length: int = 256
    normalize_steering: bool = False
    normalize_rayleigh: bool = False
    rayleigh_eps: float = 1e-6
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma_energy <= 1.0:
            raise ValueError(f"gamma_energy debe estar en (0, 1], recibido {self.gamma_energy}")
        if not 0.0 <= self.risk_percentile <= 1.0:
            raise ValueError(f"risk_percentile debe estar en [0, 1], recibido {self.risk_percentile}")
        if not 0.0 < self.tau_high < 1.0:
            raise ValueError(f"tau_high debe estar en (0, 1), recibido {self.tau_high}")
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
    """Acumula ``Σ ΔΔᵀ`` y el nº de muestras por cabeza en CPU/float64; nunca reserva VRAM."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        self.sum = torch.zeros(num_heads, head_dim, head_dim, dtype=_MATH_DTYPE)
        self.count = torch.zeros(num_heads, dtype=_MATH_DTYPE)

    @torch.no_grad()
    def update(self, delta: Tensor) -> None:
        """Muestras explícitas ``delta``: (N, H, d), en cualquier device/dtype."""
        delta = delta.detach().to(device="cpu", dtype=_MATH_DTYPE)
        self.sum.add_(torch.einsum("nhd,nhe->hde", delta, delta))
        self.count.add_(delta.shape[0])

    @torch.no_grad()
    def add(self, sums: Tensor, counts: Tensor) -> None:
        """Sumas ya reducidas: ``sums`` (H, d, d) y ``counts`` (H,)."""
        self.sum.add_(sums.detach().to(device="cpu", dtype=_MATH_DTYPE))
        self.count.add_(counts.detach().to(device="cpu", dtype=_MATH_DTYPE))

    @property
    def total(self) -> int:
        return int(self.count.sum().item())

    def finalize(self) -> Tensor:
        """Segundo momento ``E[ΔΔᵀ]``, shape (H, d, d), simétrico.

        Una cabeza sin muestras queda con Σ = 0 (su proyector será la identidad).
        """
        if self.total == 0:
            raise RuntimeError("No se ha acumulado ninguna muestra")
        sigma = self.sum / self.count.clamp_min(1.0)[:, None, None]
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
    """``R(r) = rᵀ Σ r / (‖r‖² + ε)`` (Ec. 16 de SKOP). sigma (..., d, d), r (..., d) -> (...)."""
    s = sigma.detach().to(device="cpu", dtype=_MATH_DTYPE)
    v = r.detach().to(device="cpu", dtype=_MATH_DTYPE)
    numerator = torch.einsum("...i,...ij,...j->...", v, s, v)
    return numerator / ((v * v).sum(-1) + eps)


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


#: Tolerancia al comparar la masa acumulada con ``tau_high`` (probabilidades en float32).
_MASS_TOL = 1e-6


@torch.no_grad()
def focus_set_mask(probs: Tensor, attendable: Tensor, tau_high: float) -> Tensor:
    """Focus Set de cada (cabeza, paso): el conjunto mínimo de keys cuya masa de
    atención acumulada alcanza ``tau_high`` (Algoritmo 1 de SKOP).

    Args:
        probs: probabilidades de atención, (..., T_q, T_k).
        attendable: máscara booleana de las keys atendibles, misma forma.
        tau_high: fracción de masa del Focus Set, en (0, 1).

    Las keys se ordenan por probabilidad descendente (orden estable ante empates)
    y entra cada una mientras la masa de las anteriores sea < ``tau_high``. La
    masa se mide sobre las keys atendibles de la fila.
    """
    p = torch.where(attendable, probs.float(), 0.0)
    p = p / p.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
    sorted_p, order = p.sort(dim=-1, descending=True, stable=True)
    mass_before = sorted_p.double().cumsum(-1) - sorted_p.double()
    chosen = (mass_before < tau_high - _MASS_TOL) & (sorted_p > 0)
    return torch.zeros_like(attendable).scatter(-1, order, chosen)


@torch.no_grad()
def focus_tail_second_moment(
    keys: Tensor,
    attention: Tensor,
    attention_mask: Tensor,
    tau_high: float,
    num_kv_heads: int,
) -> Tuple[Tensor, Tensor]:
    """Suma, sobre los pasos de un lote, de la media de ``Δk Δkᵀ`` en ``F_t × T_t``.

    Args:
        keys: salida de ``k_proj``, (B, T, H_kv·d).
        attention: probabilidades de atención del modelo base, (B, H, T, T)
            (fila = query, columna = key).
        attention_mask: (B, T); las queries y keys de padding se excluyen.
        tau_high: masa de atención del Focus Set (``focus_set_mask``).
        num_kv_heads: con GQA, cada cabeza de query usa las keys de su grupo.

    Returns:
        ``(sums, steps)`` en CPU/float64, por cabeza de **query**: ``sums`` (H, d, d)
        y ``steps`` (H,), el número de pasos con Focus y Tail no vacíos.
        ``sums / steps`` es ``Σ_ΔK = E_t E_{(i,j) ~ U(F_t × T_t)}[Δk Δkᵀ]``.

    Cada paso (cabeza, query ``t``) pesa lo mismo: sus pares se promedian con
    ``c_t = 1 / (|F_t|·|T_t|)``. La suma ponderada se factoriza de forma exacta como
    ``Kᵀ G K`` con ``G = diag(w) − C − Cᵀ`` (T×T), donde
    ``w_s = Σ_t c_t (F[t,s]·|T_t| + T[t,s]·|F_t|) = Σ_t F[t,s]/|F_t| + T[t,s]/|T_t|``
    y ``C = Fᵀ diag(c) T``. Así el coste es O(T³ + T²·d + T·d²) por cabeza
    (``C``, ``G K`` y ``Kᵀ (G K)``) en vez de O(P·d²) al enumerar los P pares
    (hasta O(T³·d²)). Las keys con probabilidad 0 (futuras, padding o fuera de la
    ventana deslizante) no son atendibles y no entran en ningún conjunto.
    """
    batch, num_heads, seq_len, _ = attention.shape
    group = num_heads // num_kv_heads
    head_dim = keys.shape[-1] // num_kv_heads
    device = attention.device

    valid_tok = attention_mask.to(device).bool()
    attendable = (attention > 0) & valid_tok[:, None, :, None] & valid_tok[:, None, None, :]
    focus_mask = focus_set_mask(attention, attendable, tau_high)
    focus = focus_mask.float()
    tail = (attendable & ~focus_mask).float()
    n_focus = focus.sum(-1)  # (B, H, T)
    n_tail = tail.sum(-1)
    step = (n_focus > 0) & (n_tail > 0)
    weight = step.float() / (n_focus * n_tail).clamp_min(1.0)  # c_t

    w = torch.einsum("bhts,bht->bhs", focus, weight * n_tail) + torch.einsum("bhts,bht->bhs", tail, weight * n_focus)
    cross = torch.einsum("bhts,bht,bhtu->bhsu", focus, weight, tail)
    gram = torch.diag_embed(w) - cross - cross.mT  # (B, H, T, T)

    k = keys.to(device).float().view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # (B, H_kv, T, d)
    k = k.repeat_interleave(group, dim=1)  # (B, H, T, d): keys del grupo de cada query
    kg = torch.einsum("bhtd,bhts->bhds", k, gram)
    sums = torch.einsum("bhds,bhse->hde", kg, k)
    steps = step.sum((0, 2))
    return sums.to(device="cpu", dtype=_MATH_DTYPE), steps.to(device="cpu", dtype=_MATH_DTYPE)


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


def _check_layer_ids(layers: Sequence[int], n_layers: int) -> List[int]:
    """Índices de capa ordenados y sin repetir; fuera de ``[0, n_layers)`` es un error."""
    ids = sorted({int(i) for i in layers})
    if not ids:
        raise ValueError("No hay capas que intervenir")
    outside = [i for i in ids if not 0 <= i < n_layers]
    if outside:
        raise ValueError(f"Capas fuera de rango {outside}: el modelo tiene {n_layers} capas (0–{n_layers - 1})")
    return ids


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
        self.layer_ids: List[int] = _check_layer_ids(layers, n_layers)

        self.sigma_delta_k: Dict[int, Tensor] = {}      # (H, d, d), por cabeza de query
        self.steering_vectors: Dict[int, Tensor] = {}   # r_q, (H, d)
        self.projectors: Dict[int, ProjectorResult] = {}
        self.rayleigh: Dict[int, Tensor] = {}           # (H,)
        self.risk_mask: Dict[int, Tensor] = {}          # (H,) bool
        self.risk_threshold: Optional[float] = None
        self.num_utility_samples = 0   # secuencias del utility set
        self.num_delta_steps = 0       # pasos (cabeza, query) con Focus y Tail no vacíos
        self.num_concept_pairs = 0     # pares contrastivos usados para r_q

        self._sigma_acc: Dict[int, SecondMomentAccumulator] = {}
        self._rq_acc: Dict[int, _MeanAccumulator] = {}
        self._effective: Dict[int, Tensor] = {}         # (H*d,) float64, sin α
        self._risk_sigma: Dict[int, Tensor] = {}        # Σ usada en Rayleigh (H, d, d)
        self._cast_cache: Dict[Tuple[int, torch.device, torch.dtype], Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._telemetry_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._probe: Dict[int, Tensor] = {}             # última q (post-steering) por capa, en device
        self._alpha = float(self.config.alpha)
        self._use_projection = True
        self._layer_plan: Optional[Dict[int, Tuple[float, bool]]] = None  # capa -> (alpha, proyección)
        self.metadata: Dict[str, Any] = {}
        self._direction_override: Dict[int, Tensor] = {}  # capa -> r_q en caliente (p. ej. features SAE)
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
            self.compute_risk()  # el top 20 % se calcula sobre las capas activas
        self._cast_cache.clear()
        if self.is_attached:
            self.attach()
        return self

    # ------------------------------------------------ direcciones externas
    @property
    def hidden_size(self) -> int:
        config = self.model.config
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()
        return int(config.hidden_size)

    @torch.no_grad()
    def residual_to_query(self, layer: int, direction: Tensor) -> Tensor:
        """Mapea una dirección del residual stream al espacio de queries de ``layer``.

        ``r_q = q_proj(input_layernorm(v))`` (p. ej. ``v`` = decoder de una feature de
        un SAE residual) es una **aproximación** del cambio de la query cuando se
        empuja el residual ``h`` de un token en la dirección ``v``:

        - RMSNorm se linealiza en ``v`` y no en ``h``. Su Jacobiano real,
          ``J(h) v ∝ g ⊙ (v − h (h·v) / ‖h‖²) / rms(h)``, depende del token: la
          dirección coincide sólo si ``v ⊥ h``, y se desvía más cuanto más se alinea
          ``v`` con el residual (justo el caso de una feature activa). La escala
          tampoco es la real: aquí la fija ``|v|``, en el modelo ``rms(h)``.
        - El vector vive en el espacio de queries **previo a RoPE**. RoPE lo rota con
          la posición de cada query, de modo que su efecto en el score con la key
          ``i`` depende del desfase ``i − t`` (ver la nota sobre RoPE del módulo).

        Llama a ``.forward`` de los módulos para no disparar los hooks de
        steering/telemetría.

        Returns: (H, d) en float64/CPU.
        """
        g = self.geometry
        vec = direction.detach().reshape(-1)
        if vec.numel() != self.hidden_size:
            raise ValueError(
                f"La dirección tiene {vec.numel()} componentes y el residual del modelo {self.hidden_size}"
            )
        if not torch.isfinite(vec).all() or vec.abs().max() == 0:
            raise ValueError("La dirección debe ser finita y no nula")
        block = self.decoder_layers[layer]
        q_proj = block.self_attn.q_proj
        norm = getattr(block, "input_layernorm", None)
        ref = getattr(norm, "weight", None)
        if not isinstance(ref, Tensor) or not ref.is_floating_point():
            ref = getattr(q_proj, "weight", None)  # sin norm (o pesos cuantizados): usa q_proj
        device = ref.device if isinstance(ref, Tensor) else self._input_device()
        dtype = ref.dtype if isinstance(ref, Tensor) and ref.is_floating_point() else torch.float32
        x = vec.to(device=device, dtype=dtype).view(1, 1, -1)
        if norm is not None:
            x = norm.forward(x)
        q = q_proj.forward(x)
        return q.reshape(g.num_heads, g.head_dim).to(device="cpu", dtype=_MATH_DTYPE)

    def steering_from_residual(self, direction: Tensor, layers: Optional[Sequence[int]] = None) -> Dict[int, Tensor]:
        """``{capa: r_q}`` de una misma dirección residual; útil para ``fit(steering_vectors=...)``."""
        return {l: self.residual_to_query(l, direction) for l in (self.layer_ids if layers is None else layers)}

    @property
    def layer_directions(self) -> Dict[int, Tensor]:
        return dict(self._direction_override)

    def direction_for(self, layer: int) -> Tensor:
        """``r_q`` efectivo de la capa: la dirección inyectada en caliente o la calibrada."""
        if layer in self._direction_override:
            return self._direction_override[layer]
        return self.steering_vectors[layer]

    def set_layer_directions(self, directions: Optional[Mapping[int, Tensor]]) -> "SKOPEngine":
        """Sustituye en caliente ``r_q`` en algunas capas; ``None`` vuelve a las calibradas.

        Las direcciones (H, d) suelen venir de ``residual_to_query`` (features de un
        SAE). El ranking global de Rayleigh se recalcula con las direcciones efectivas,
        de modo que P_ΔK se aplica sólo a las cabezas de alto riesgo *para esas
        direcciones*. Los ``r_q`` calibrados no se modifican ni se guardan con ellas.
        """
        g = self.geometry
        overrides: Dict[int, Tensor] = {}
        for l, v in (directions or {}).items():
            if l not in self.layer_ids:
                raise KeyError(f"La capa {l} no está en layer_ids")
            d = v.detach().to("cpu", _MATH_DTYPE).reshape(g.num_heads, g.head_dim)
            if not torch.isfinite(d).all():
                raise ValueError(f"Dirección no finita para la capa {l}")
            overrides[int(l)] = d
        self._direction_override = overrides
        if self.projectors and all(l in self.steering_vectors or l in overrides for l in self.layer_ids):
            self.compute_risk()
        return self

    def circuit_state(self) -> Dict[str, Any]:
        """Copia ligera del estado mutable del circuito, para deshacer un cambio a medias.

        Copia los diccionarios, no los tensores: el motor nunca modifica un tensor en
        su sitio, siempre los sustituye.
        """
        return {
            "config": replace(self.config),
            "alpha": self._alpha,
            "use_projection": self._use_projection,
            "layer_plan": None if self._layer_plan is None else dict(self._layer_plan),
            "directions": dict(self._direction_override),
            "projectors": dict(self.projectors),
            "rayleigh": dict(self.rayleigh),
            "risk_mask": dict(self.risk_mask),
            "risk_threshold": self.risk_threshold,
            "effective": dict(self._effective),
            "risk_sigma": dict(self._risk_sigma),
            "attached": self.is_attached,
        }

    def restore_circuit_state(self, state: Mapping[str, Any]) -> None:
        """Vuelve al estado de ``circuit_state()``, hooks incluidos."""
        self.config = state["config"]
        self._alpha = state["alpha"]
        self._use_projection = state["use_projection"]
        self._layer_plan = state["layer_plan"]
        self._direction_override = dict(state["directions"])
        self.projectors = dict(state["projectors"])
        self.rayleigh = dict(state["rayleigh"])
        self.risk_mask = dict(state["risk_mask"])
        self.risk_threshold = state["risk_threshold"]
        self._effective = dict(state["effective"])
        self._risk_sigma = dict(state["risk_sigma"])
        self._cast_cache.clear()
        self.detach()
        if state["attached"]:
            self.attach()

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
        self._sigma_acc = {l: SecondMomentAccumulator(g.num_heads, g.head_dim) for l in self.layer_ids}
        self._rq_acc = {l: _MeanAccumulator(g.num_heads, g.head_dim) for l in self.layer_ids}
        self._utility_seen = 0

    @torch.no_grad()
    def calibrate(
        self,
        utility_texts: Sequence[str],
        positive_prompts: Optional[Sequence[str]] = None,
        negative_prompts: Optional[Sequence[str]] = None,
        reset: bool = True,
    ) -> "SKOPEngine":
        """Calibración offline completa.

        ``utility_texts`` estima Σ_ΔK (Focus/Tail); los pares de concepto, si se
        dan, estiman ``r_q``. Sin pares, ``r_q`` debe pasarse a ``fit()``.
        """
        if self.tokenizer is None:
            raise RuntimeError("calibrate() necesita un tokenizer; usa accumulate_*_batch()")
        if not utility_texts:
            raise ValueError("utility_texts está vacío")
        if (positive_prompts is None) != (negative_prompts is None):
            raise ValueError("positive_prompts y negative_prompts van juntos")
        if positive_prompts is not None and len(positive_prompts) != len(negative_prompts):
            raise ValueError("positive_prompts y negative_prompts deben tener la misma longitud")
        if getattr(self.tokenizer, "pad_token", "") is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Llama-3 no trae pad
        if reset:
            self.reset_calibration()

        bs = self.config.batch_size
        for start in range(0, len(utility_texts), bs):
            self.accumulate_utility_batch(self._encode(utility_texts[start : start + bs]))
        if positive_prompts is not None:
            for start in range(0, len(positive_prompts), bs):
                self.accumulate_concept_batch(
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
    def accumulate_utility_batch(self, inputs: Mapping[str, Tensor]) -> None:
        """Acumula Σ_ΔK (Focus/Tail) de un lote tokenizado del utility set.

        Por capa, el hook de ``k_proj`` guarda las keys en el device y el hook del
        módulo de atención, con los pesos reales, reduce los pasos Focus × Tail a
        una matriz (H, d, d) que baja a CPU. Nunca hay más de una capa de keys y
        atención en memoria a la vez.
        """
        input_ids, mask = self._prepare(inputs)
        g, tau = self.geometry, self.config.tau_high
        keys: Dict[int, Tensor] = {}
        results: Dict[int, Tuple[Tensor, Tensor]] = {}

        def key_hook(layer: int):
            def hook(module: nn.Module, args: Any, output: Tensor) -> None:
                keys[layer] = output.detach()

            return hook

        def attention_hook(layer: int):
            def hook(module: nn.Module, args: Any, output: Any) -> None:
                weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if not isinstance(weights, Tensor):
                    raise RuntimeError(
                        "El módulo de atención no devuelve pesos: carga el modelo con "
                        "attn_implementation='eager' (SDPA/Flash no los exponen) y usa "
                        "transformers>=4.48 (antes sólo los devolvía con output_attentions)"
                    )
                results[layer] = focus_tail_second_moment(keys.pop(layer), weights, mask, tau, g.num_kv_heads)

            return hook

        handles = []
        for l in self.layer_ids:
            attn = self.decoder_layers[l].self_attn
            handles.append(attn.k_proj.register_forward_hook(key_hook(l)))
            handles.append(attn.register_forward_hook(attention_hook(l)))
        self._forward_until_last_layer(input_ids, mask, handles)

        missing = [l for l in self.layer_ids if l not in results]
        if missing:
            raise RuntimeError(f"No se capturó la atención en las capas {missing}")
        # Se aplica sólo si todas las capas tuvieron éxito: el lote entra entero o no entra.
        for l, (sums, steps) in results.items():
            self._sigma_acc[l].add(sums, steps)
        self._utility_seen += int(input_ids.shape[0])

    @torch.no_grad()
    def accumulate_concept_batch(
        self, positive_inputs: Mapping[str, Tensor], negative_inputs: Mapping[str, Tensor]
    ) -> None:
        """Acumula ``r_q = E[q_pos − q_neg]`` de un lote de pares de concepto tokenizados."""
        q_pos = self._capture_queries(positive_inputs)
        q_neg = self._capture_queries(negative_inputs)
        for l in self.layer_ids:
            self._rq_acc[l].update(q_pos[l].to(_MATH_DTYPE) - q_neg[l].to(_MATH_DTYPE))

    def _capture_queries(self, inputs: Mapping[str, Tensor]) -> Dict[int, Tensor]:
        """Un forward; q agregada por secuencia y capa, ya en CPU (float32)."""
        input_ids, mask = self._prepare(inputs)
        g, mode = self.geometry, self.config.token_aggregation
        captured: Dict[int, Tensor] = {}

        def make_hook(layer: int):
            def hook(module: nn.Module, args: Any, output: Tensor) -> None:
                captured[layer] = _aggregate_tokens(output, mask, g.num_heads, g.head_dim, mode).cpu()

            return hook

        handles = [self.decoder_layers[l].self_attn.q_proj.register_forward_hook(make_hook(l)) for l in self.layer_ids]
        self._forward_until_last_layer(input_ids, mask, handles)
        missing = [l for l in self.layer_ids if l not in captured]
        if missing:
            raise RuntimeError(f"No se capturaron queries en las capas {missing}")
        return captured

    def _prepare(self, inputs: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        if self.is_attached:
            raise RuntimeError("Llama a detach() antes de calibrar: el steering contaminaría la calibración")
        device = self._input_device()
        input_ids = inputs["input_ids"].to(device)
        mask = inputs.get("attention_mask")
        return input_ids, (torch.ones_like(input_ids) if mask is None else mask.to(device))

    def _forward_until_last_layer(self, input_ids: Tensor, mask: Tensor, handles: List[Any]) -> None:
        """Forward con los hooks dados, cortado tras la última capa calibrada; siempre los retira."""
        try:
            handles.append(self.decoder_layers[self.layer_ids[-1]].register_forward_hook(_raise_stop))
            try:
                self.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
            except _StopForward:
                pass
        finally:
            for h in handles:
                h.remove()

    def _input_device(self) -> torch.device:
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings().weight.device
        return next(self.model.parameters()).device

    def finalize_calibration(self) -> "SKOPEngine":
        """Convierte los acumuladores en Σ_ΔK y, si hubo pares de concepto, en ``r_q``."""
        first = self.layer_ids[0]
        if self._sigma_acc[first].total == 0:
            raise RuntimeError(
                "Σ_ΔK sin pasos con Focus y Tail no vacíos: añade textos de utilidad más largos "
                "(o baja tau_high: si el Focus Set necesita todas las keys, el Tail queda vacío)"
            )
        for l in self.layer_ids:
            self.sigma_delta_k[l] = self._sigma_acc[l].finalize()
        self.num_utility_samples = self._utility_seen
        self.num_delta_steps = sum(self._sigma_acc[l].total for l in self.layer_ids)
        if self._rq_acc[first].count > 0:
            for l in self.layer_ids:
                r = self._rq_acc[l].finalize()
                if self.config.normalize_steering:
                    r = r / r.norm(dim=-1, keepdim=True).clamp_min(self.config.eps)
                self.steering_vectors[l] = r
            self.num_concept_pairs = self._rq_acc[first].count
        return self

    def set_steering_vectors(self, vectors: Mapping[int, Tensor]) -> "SKOPEngine":
        """Sustituye ``r_q`` por vectores externos, shape (H, d) o (H*d,) por capa."""
        self._store_steering_vectors(vectors)
        if self.projectors:
            self.compute_risk()
        return self

    def _store_steering_vectors(self, vectors: Mapping[int, Tensor]) -> None:
        g = self.geometry
        for l, v in vectors.items():
            if l not in self.layer_ids:
                raise KeyError(f"La capa {l} no está en layer_ids")
            self.steering_vectors[l] = v.detach().to("cpu", _MATH_DTYPE).reshape(g.num_heads, g.head_dim)

    # ------------------------------------------------------------ ajuste SKOP
    def fit(self, steering_vectors: Optional[Mapping[int, Tensor]] = None) -> "SKOPEngine":
        """Proyectores + filtrado de Rayleigh + vectores efectivos.

        ``steering_vectors`` (capa -> (H, d)) carga una dirección conceptual ``r_q``
        externa; si se omite, se usa la estimada con pares de concepto.
        """
        if steering_vectors is not None:
            self._store_steering_vectors(steering_vectors)
        missing = [l for l in self.layer_ids if l not in self.steering_vectors and l not in self._direction_override]
        if missing:
            raise RuntimeError(
                f"Falta r_q en las capas {missing}: pásalo a fit(steering_vectors=...) "
                "o calibra con pares de concepto"
            )
        return self.fit_projectors().compute_risk()

    def fit_projectors(self) -> "SKOPEngine":
        if not self.sigma_delta_k:
            raise RuntimeError("Ejecuta calibrate() con el utility set antes de fit()")
        for l in self.layer_ids:
            self.projectors[l] = compute_orthogonal_projector(
                self.sigma_delta_k[l], self.config.gamma_energy, self.config.eps
            )
        return self

    def compute_risk(self) -> "SKOPEngine":
        """Cociente de Rayleigh por cabeza de query y máscara top-(1-p).

        El percentil se calcula sobre las cabezas de ``active_layers``: todas las
        capas calibradas en modo global (el ranking global del paper) o sólo las del
        plan por capa. Las capas sin steering no compiten por el 20 % ni se marcan.
        """
        if not self.projectors or not (self.steering_vectors or self._direction_override):
            raise RuntimeError("Faltan proyectores o vectores de steering")
        for l in self.layer_ids:
            sigma = self.sigma_delta_k[l]
            if self.config.normalize_rayleigh:
                sigma = trace_normalize(sigma, self.config.rayleigh_eps)
            self._risk_sigma[l] = sigma
            self.rayleigh[l] = rayleigh_quotient(self._risk_sigma[l], self.direction_for(l), self.config.eps)
        active = self.active_layers
        scores = torch.stack([self.rayleigh[l] for l in active])  # (L_activas, H)
        _, self.risk_threshold = select_risky_heads(scores, self.config.risk_percentile)
        for l in self.layer_ids:
            self.risk_mask[l] = (
                self.rayleigh[l] > self.risk_threshold if l in active else torch.zeros_like(self.rayleigh[l], dtype=torch.bool)
            )
        self._build_effective_vectors()
        return self

    def _build_effective_vectors(self) -> None:
        for l in self.layer_ids:
            r = self.direction_for(l)
            projected = torch.einsum("hij,hj->hi", self.projectors[l].projector, r)
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
        """Cociente de Rayleigh en vivo de la query actual (ya steerada), por capa: (B, H)."""
        return self.collect_telemetry_pair(clear)[0]

    @torch.no_grad()
    def collect_telemetry_pair(self, clear: bool = True) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
        """``R(q)`` de la query actual con steering y ``R(q − δ)`` de la misma query sin
        él, por capa: (B, H) float64 cada uno.

        ``δ`` es exactamente el vector que sumó el hook de steering de esa capa (en las
        capas sin steering, ``q − δ = q``). Las dos usan la misma Σ que el ranking, así
        que su diferencia mide cuánto mueve la inyección de esa capa a cada cabeza en el
        subespacio Focus/Tail. El efecto de las capas anteriores, que llega por el
        residual stream, sigue dentro de ambas. ``rayleigh`` y ``risk_threshold`` se
        refieren a las direcciones de steering, no a queries: no son comparables con
        estos valores.
        """
        g = self.geometry
        steered_layers = set(self.active_layers) if self.is_attached else set()
        live: Dict[int, Tensor] = {}
        base: Dict[int, Tensor] = {}
        for l, q in self._probe.items():
            qh = q.to(device="cpu", dtype=_MATH_DTYPE).reshape(q.shape[0], g.num_heads, g.head_dim)
            sigma = self._risk_sigma[l].expand(qh.shape[0], -1, -1, -1)
            live[l] = rayleigh_quotient(sigma, qh, self.config.eps)
            if l in steered_layers:
                delta = self._steering_tensor(l, q.device, q.dtype).to(device="cpu", dtype=_MATH_DTYPE)
                base[l] = rayleigh_quotient(sigma, qh - delta.view(1, g.num_heads, g.head_dim), self.config.eps)
            else:
                base[l] = live[l]
        if clear:
            self._probe.clear()
        return live, base

    # ------------------------------------------------------------ inspección
    def summary(self) -> Dict[str, Any]:
        return {
            "num_utility_samples": self.num_utility_samples,
            "num_delta_steps": self.num_delta_steps,
            "num_concept_pairs": self.num_concept_pairs,
            "tau_high": self.config.tau_high,
            "gamma_energy": self.config.gamma_energy,
            "risk_percentile": self.config.risk_percentile,
            "risk_threshold": self.risk_threshold,
            "normalize_rayleigh": self.config.normalize_rayleigh,
            "use_projection": self._use_projection,
            "alpha": self._alpha,
            "layer_plan": self.layer_plan,
            "layers": {
                l: {
                    "rank_per_head": self.projectors[l].rank.tolist(),
                    "rayleigh": self.rayleigh[l].tolist(),
                    "risky_heads": self.risk_mask[l].nonzero().flatten().tolist(),
                }
                for l in self.layer_ids
                if l in self.projectors and l in self.rayleigh
            },
        }

    # ---------------------------------------------------------- persistencia
    def state_dict(self, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Artefacto completo: Σ_ΔK y ``r_q`` en float64 (P se recalcula al cargar)."""
        return {
            "format": ARTIFACT_FORMAT,
            "layer_ids": list(self.layer_ids),
            "geometry": vars(self.geometry).copy(),
            "tau_high": self.config.tau_high,
            "fit_config": {k: getattr(self.config, k) for k in FIT_CONFIG_KEYS},
            "num_utility_samples": self.num_utility_samples,
            "num_delta_steps": self.num_delta_steps,
            "num_concept_pairs": self.num_concept_pairs,
            "sigma_delta_k": dict(self.sigma_delta_k),
            "steering_vectors": dict(self.steering_vectors),
            "meta": dict(metadata or {}),
        }

    def load_state_dict(
        self, state: Mapping[str, Any], fit: bool = True, use_saved_fit_config: bool = True
    ) -> "SKOPEngine":
        """Carga un artefacto de ``state_dict``/``save``.

        Con ``use_saved_fit_config`` (por defecto), γ, el percentil de riesgo y la
        normalización de Rayleigh pasan a ser los del artefacto, es decir, los de la
        calibración (p. ej. ``latentflow-calibrate --gamma 0.8``). Los artefactos v3
        anteriores, sin ``fit_config``, conservan la configuración del motor.
        """
        if state.get("format") != ARTIFACT_FORMAT:
            raise ValueError(
                f"Artefacto de calibración con formato {state.get('format')!r}; se esperaba "
                f"{ARTIFACT_FORMAT!r}. Regenéralo con `latentflow-calibrate`."
            )
        if state["geometry"] != vars(self.geometry):
            raise ValueError("La calibración pertenece a un modelo con otra geometría de atención")
        saved_fit = state.get("fit_config") or {}
        if use_saved_fit_config and saved_fit:
            # replace() pasa por __post_init__: un valor fuera de rango no se acepta.
            self.config = replace(self.config, **{k: saved_fit[k] for k in FIT_CONFIG_KEYS if k in saved_fit})
        saved = _check_layer_ids(state["layer_ids"], len(self.decoder_layers))
        if self.config.layers is None:
            # Sin capas pedidas, el motor adopta las del artefacto (p. ej. `--layers 0,2`).
            layers = saved
        else:
            missing = sorted(set(self.layer_ids) - set(saved))
            if missing:
                raise ValueError(f"La calibración no cubre las capas pedidas {missing} (cubre {saved})")
            layers = self.layer_ids
        was_attached = self.is_attached
        self._set_layer_ids(layers)
        sigma = {int(l): t for l, t in state["sigma_delta_k"].items()}
        vectors = {int(l): t for l, t in state["steering_vectors"].items()}
        self.sigma_delta_k = {l: sigma[l].to(_MATH_DTYPE) for l in layers}
        self.steering_vectors = {l: vectors[l].to(_MATH_DTYPE) for l in layers if l in vectors}
        self.num_utility_samples = int(state["num_utility_samples"])
        self.num_delta_steps = int(state["num_delta_steps"])
        self.num_concept_pairs = int(state["num_concept_pairs"])
        self.metadata = dict(state.get("meta", {}))
        if fit:
            self.fit()
            if was_attached:
                self.attach()
        return self

    def _set_layer_ids(self, layers: Sequence[int]) -> None:
        """Cambia las capas del motor y descarta todo lo calculado sobre las anteriores."""
        layers = list(layers)
        if layers == self.layer_ids:
            return
        self.detach()
        self.detach_telemetry()
        self.layer_ids = layers
        for table in (self.sigma_delta_k, self.steering_vectors, self.projectors, self.rayleigh,
                      self.risk_mask, self._effective, self._risk_sigma):
            table.clear()
        self.risk_threshold = None
        self._cast_cache.clear()
        if self._layer_plan is not None:
            self._layer_plan = {l: v for l, v in self._layer_plan.items() if l in layers} or None
        self._direction_override = {l: v for l, v in self._direction_override.items() if l in layers}
        self.reset_calibration()

    def save(self, path: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        torch.save(self.state_dict(metadata), path)

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
    quantize: bool = True,
):
    """Carga un modelo HF cuantizado con bitsandbytes para una GPU de ~8 GB.

    ``quantize=False`` carga el modelo sin cuantizar en ``compute_dtype`` (CPU o GPU
    sin bitsandbytes). La calibración de Σ_ΔK necesita ``attn_implementation="eager"``.

    Gemma-2-2B-IT cabe holgado en 4 bit; Llama-3.1-8B en 4 bit ocupa ~5.5 GB.
    Usa ``compute_dtype=torch.float16`` en GPUs sin bfloat16 (pre-Ampere), pero
    ten en cuenta que Gemma-2 es propensa a desbordar en float16. Gemma-2
    recomienda ``attn_implementation="eager"`` por el soft-capping de logits.
    """
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not quantize:
        quantization = None
    elif load_in_4bit:
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
