"""
latentflow.core.sae
===================

Resolución de *features* de Sparse Autoencoders (Gemma Scope / Neuronpedia) a su
vector decoder en el residual stream, sin acceso a red.

Un ID de Neuronpedia como ``gemma-2-2b/20-gemmascope-res-16k/12082`` identifica la
fila 12082 de ``W_dec`` del SAE residual (``res``) de la capa 20, ancho 16k. Ese
vector vive en el residual stream (``d_model``), no en el espacio de queries: el
``SKOPEngine`` lo mapea con ``residual_to_query()`` antes de usarlo como ``r_q``.

Orden de búsqueda de ``GemmaScopeResolver`` (``root`` = ``LATENTFLOW_SAE_DIR``):

1. Fichero estructurado: ``<root>/<modelo>/<sae_id>.{safetensors,npz,pt}`` o
   ``<root>/<modelo>/<sae_id>/params.{safetensors,npz}``, con una matriz ``W_dec``
   en cualquier orientación ((d_sae, d_model) como Gemma Scope, o (d_model, d_sae)).
2. Árbol de Gemma Scope bajo ``root``:
   ``[<root>/gemma-scope-2b-pt-res-canonical/]layer_20/width_16k/canonical/params.npz``
   (o un único ``average_l0_*``).
3. Caché local de Hugging Face (``huggingface_hub.try_to_load_from_cache``, sin
   red) del repo ``google/gemma-scope-<tamaño>-pt-<sitio>-canonical``.

Para poblarla en una máquina con red::

    hf download google/gemma-scope-2b-pt-res-canonical \\
        --include "layer_20/width_16k/canonical/*"

``HashFeatureResolver`` (dev backend) genera un vector pseudoaleatorio determinista
a partir del sha256 del ID: el mismo ID da el mismo vector en cualquier proceso
(``hash()`` de Python no sirve: se aleatoriza por proceso).
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import torch
from torch import Tensor

__all__ = [
    "FeatureError",
    "FeatureNotFoundError",
    "FeatureRef",
    "FeatureResolver",
    "GemmaScopeResolver",
    "HashFeatureResolver",
    "ResolvedFeature",
]

# Sitios de Gemma Scope cuyo decoder escribe en el residual stream.
RESIDUAL_SITES = ("res", "mlp")

_GEMMA_SCOPE_ID = re.compile(
    r"^(?P<model>[\w.-]+)/(?P<layer>\d+)-gemmascope-(?P<site>res|mlp|att)-(?P<width>\d+[km])/(?P<index>\d+)$"
)
_GENERIC_ID = re.compile(r"^(?P<model>[\w.-]+)/(?P<sae>[\w.-]+)/(?P<index>\d+)$")


class FeatureError(ValueError):
    """Feature no válida o incompatible con el modelo (mensaje para el usuario)."""


class FeatureNotFoundError(FeatureError):
    """No hay pesos locales para el SAE de la feature."""


@dataclass(frozen=True)
class FeatureRef:
    """ID ``modelo/sae_id/índice`` (formato de Neuronpedia)."""

    raw: str
    model: str
    sae_id: str
    index: int
    layer: Optional[int] = None   # capa del SAE (sólo IDs de Gemma Scope)
    site: Optional[str] = None    # res | mlp | att
    width: Optional[str] = None   # 16k, 65k, ...

    @classmethod
    def parse(cls, feature_id: str) -> "FeatureRef":
        raw = feature_id.strip()
        m = _GEMMA_SCOPE_ID.match(raw)
        if m:
            sae_id = raw.split("/")[1]
            return cls(raw, m["model"], sae_id, int(m["index"]), int(m["layer"]), m["site"], m["width"])
        m = _GENERIC_ID.match(raw)
        if m:
            return cls(raw, m["model"], m["sae"], int(m["index"]))
        raise FeatureError(f"Feature ID '{feature_id}' no tiene la forma modelo/sae/índice")

    @property
    def model_size(self) -> Optional[str]:
        m = re.search(r"gemma-2-(\d+b)", self.model)
        return m.group(1) if m else None


@dataclass(frozen=True)
class ResolvedFeature:
    feature_id: str
    vector: Tensor          # (d_model,) float32 en CPU, dirección del residual stream
    source: str             # "gemma-scope:<ruta>" o "dev-hash"
    layer: Optional[int] = None
    site: Optional[str] = None


class FeatureResolver(Protocol):
    def resolve(self, feature_id: str, d_model: int) -> ResolvedFeature: ...

    def describe(self) -> Dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# Dev backend: vector pseudoaleatorio determinista
# --------------------------------------------------------------------------- #
class HashFeatureResolver:
    """Simula features: vector unitario derivado del sha256 del ID (offline, determinista)."""

    def resolve(self, feature_id: str, d_model: int) -> ResolvedFeature:
        raw = feature_id.strip()
        if not raw:
            raise FeatureError("Feature ID vacío")
        ref = None
        try:
            ref = FeatureRef.parse(raw)
        except FeatureError:
            pass  # en dev se acepta cualquier ID no vacío
        seed = int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big")
        vector = torch.randn(d_model, generator=torch.Generator().manual_seed(seed))
        return ResolvedFeature(raw, vector / vector.norm(), "dev-hash", ref.layer if ref else None, ref.site if ref else None)

    def describe(self) -> Dict[str, Any]:
        return {"kind": "dev-hash"}


# --------------------------------------------------------------------------- #
# Gemma Scope (pesos locales)
# --------------------------------------------------------------------------- #
class GemmaScopeResolver:
    """Lee ``W_dec[índice]`` de los pesos locales de un SAE (ver orden de búsqueda en el módulo)."""

    def __init__(
        self,
        root: Optional[str] = None,
        use_hf_cache: bool = True,
        max_cached_saes: int = 2,
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        self.root = Path(root).expanduser() if root else None
        self.use_hf_cache = use_hf_cache
        self.hf_cache_dir = hf_cache_dir  # None: la caché por defecto de huggingface_hub
        self.max_cached_saes = max_cached_saes
        self._decoders: "OrderedDict[Path, Any]" = OrderedDict()   # LRU de matrices W_dec (npz/pt)
        self._vectors: Dict[str, ResolvedFeature] = {}
        self._lock = threading.Lock()

    def describe(self) -> Dict[str, Any]:
        return {"kind": "gemma-scope", "root": str(self.root) if self.root else None, "hf_cache": self.use_hf_cache}

    def resolve(self, feature_id: str, d_model: int) -> ResolvedFeature:
        ref = FeatureRef.parse(feature_id)
        if ref.site is not None and ref.site not in RESIDUAL_SITES:
            raise FeatureError(
                f"{ref.raw}: los SAEs de atención ('{ref.site}') codifican salidas de cabezas, no el residual "
                "stream; usa features 'res' o 'mlp'"
            )
        key = f"{ref.raw}@{d_model}"
        with self._lock:
            if key in self._vectors:
                return self._vectors[key]
            path = self._find(ref)
            vector = self._read_row(path, ref, d_model)
            resolved = ResolvedFeature(ref.raw, vector, f"gemma-scope:{path}", ref.layer, ref.site)
            self._vectors[key] = resolved
            return resolved

    # ---------------------------------------------------------------- búsqueda
    def _candidates(self, ref: FeatureRef) -> List[Path]:
        out: List[Path] = []
        if self.root is not None:
            base = self.root / ref.model
            out += [base / f"{ref.sae_id}{ext}" for ext in (".safetensors", ".npz", ".pt")]
            out += [base / ref.sae_id / f"params{ext}" for ext in (".safetensors", ".npz")]
            if ref.layer is not None and ref.model_size:
                rel = Path(f"layer_{ref.layer}") / f"width_{ref.width}"
                for tree in (self.root / f"gemma-scope-{ref.model_size}-pt-{ref.site}-canonical", self.root):
                    out.append(tree / rel / "canonical" / "params.npz")
                    variants = sorted((tree / rel).glob("average_l0_*/params.npz"))
                    if len(variants) == 1:
                        out.append(variants[0])
        return out

    def _find(self, ref: FeatureRef) -> Path:
        candidates = self._candidates(ref)
        for path in candidates:
            if path.is_file():
                return path
        hf_target = None
        if self.use_hf_cache and ref.layer is not None and ref.model_size:
            repo = f"google/gemma-scope-{ref.model_size}-pt-{ref.site}-canonical"
            filename = f"layer_{ref.layer}/width_{ref.width}/canonical/params.npz"
            hf_target = f"{repo}:{filename}"
            try:
                from huggingface_hub import try_to_load_from_cache

                cached = try_to_load_from_cache(repo, filename, cache_dir=self.hf_cache_dir)
                if isinstance(cached, str):
                    return Path(cached)
            except ImportError:
                pass
        searched = [str(p) for p in candidates] + ([f"caché HF {hf_target}"] if hf_target else [])
        hint = f" Descarga: hf download {hf_target.split(':')[0]} --include \"{hf_target.split(':')[1].rsplit('/', 1)[0]}/*\"" if hf_target else ""
        where = "; ".join(searched) if searched else "define LATENTFLOW_SAE_DIR"
        raise FeatureNotFoundError(f"{ref.raw}: no hay pesos locales del SAE '{ref.sae_id}' (buscado en: {where}).{hint}")

    # ---------------------------------------------------------------- lectura
    def _read_row(self, path: Path, ref: FeatureRef, d_model: int) -> Tensor:
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(path), framework="pt") as f:
                if "W_dec" not in f.keys():
                    raise FeatureError(f"{path}: no contiene 'W_dec'")
                sl = f.get_slice("W_dec")
                rows, cols = sl.get_shape()
                axis = self._feature_axis(rows, cols, d_model, path)
                n = rows if axis == 0 else cols
                self._check_index(ref, n, path)
                vec = sl[ref.index] if axis == 0 else sl[:, ref.index]
                return torch.as_tensor(vec).reshape(-1).float().clone()
        w_dec = self._load_decoder(path)
        rows, cols = w_dec.shape
        axis = self._feature_axis(rows, cols, d_model, path)
        self._check_index(ref, rows if axis == 0 else cols, path)
        vec = w_dec[ref.index] if axis == 0 else w_dec[:, ref.index]
        return torch.as_tensor(vec).reshape(-1).float().clone()

    def _load_decoder(self, path: Path) -> Any:
        if path in self._decoders:
            self._decoders.move_to_end(path)
            return self._decoders[path]
        if path.suffix == ".npz":
            import numpy as np

            with np.load(path) as data:
                if "W_dec" not in data.files:
                    raise FeatureError(f"{path}: no contiene 'W_dec'")
                w_dec = data["W_dec"]
        else:
            obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            w_dec = obj.get("W_dec") if isinstance(obj, dict) else obj
            if not isinstance(w_dec, Tensor) or w_dec.dim() != 2:
                raise FeatureError(f"{path}: se esperaba un tensor 'W_dec' 2D")
        self._decoders[path] = w_dec
        while len(self._decoders) > self.max_cached_saes:
            self._decoders.popitem(last=False)
        return w_dec

    @staticmethod
    def _feature_axis(rows: int, cols: int, d_model: int, path: Path) -> int:
        """Eje que indexa features: Gemma Scope guarda (d_sae, d_model); se acepta la traspuesta."""
        if cols == d_model and rows != d_model:
            return 0
        if rows == d_model and cols != d_model:
            return 1
        if rows == cols == d_model:
            return 0
        raise FeatureError(
            f"{path}: W_dec tiene forma ({rows}, {cols}) y el residual del modelo es d_model={d_model}; "
            "el SAE no corresponde a este modelo"
        )

    @staticmethod
    def _check_index(ref: FeatureRef, n_features: int, path: Path) -> None:
        if not 0 <= ref.index < n_features:
            raise FeatureError(f"{ref.raw}: índice fuera de rango (el SAE de {path.name} tiene {n_features} features)")
