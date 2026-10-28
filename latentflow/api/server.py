"""
latentflow.api.server
=====================

Servidor FastAPI de BrainSteer: streaming token a token por WebSocket con
telemetría SKOP en vivo y hot-swapping del circuito de steering.

Arranque::

    LATENTFLOW_MODEL_ID=google/gemma-2-2b-it \\
    LATENTFLOW_CALIBRATION=calib/gemma2_skop.pt \\
    latentflow-server                      # o: uvicorn latentflow.api.server:app

Importar el módulo no crea la app ni lee el entorno; ``latentflow-server`` valida
``LATENTFLOW_*`` al arrancar y sale con código 2 y un mensaje claro si algo falla.

Protocolo ``/ws/stream``
------------------------
Cliente -> servidor (JSON, campo ``action``)::

    {"action": "generate", "prompt": "...", "max_new_tokens": 128, "temperature": 0.0,
     "top_p": 1.0, "chat": true, "telemetry": "risky", "request_id": "opcional"}
    {"action": "update", "alpha": 6.0, "steering": true, "skop_filter": false,
     "gamma_energy": 0.9, "risk_percentile": 0.8}
    {"action": "update", "steering": true,
     "layers": [{"layer": 6, "alpha": 2.5, "mode": "orthogonal"},
                {"layer": 12, "alpha": -1.0, "mode": "additive"}]}
    {"action": "update", "reset_layers": true}
    {"action": "update", "steering": true,
     "layers": [{"layer": 21, "alpha": 1.0, "mode": "orthogonal",
                 "features": [{"feature_id": "gemma-2-2b/20-gemmascope-res-16k/12082", "alpha": 2.5},
                              {"feature_id": null, "alpha": -1.0}]}]}
    {"action": "cancel"}
    {"action": "calibrate", "utility": ["..."], "positive": ["..."], "negative": ["..."]}
    {"action": "status"}
    {"action": "ping"}

Servidor -> cliente (JSON, campo ``type``)::

    ready, accepted, start, token, end, update_accepted, circuit, calibrated,
    status, pong, error

Features de SAE (Gemma Scope)
-----------------------------
Con ``features``, la dirección de la capa es ``alpha · Σ_f alpha_f · r_q(f)``, donde
``r_q(f) = q_proj(input_layernorm(W_dec[f]))`` para una feature (el decoder se lee
de los pesos locales, ver ``latentflow.core.sae``) y el ``r_q`` calibrado para
``feature_id: null``. El riesgo de Rayleigh se recalcula con esa dirección, así que
P_ΔK sólo se aplica a las cabezas de alto riesgo para ella. La lectura del decoder
ocurre fuera del worker; el mapeo, en la frontera de token.

Seguridad
---------
El handshake de ``/ws/stream`` valida el header ``Origin`` contra una lista blanca
(``LATENTFLOW_ALLOWED_ORIGINS``, separada por comas; por defecto el frontend local
de Vite y Tauri). Un origen no permitido recibe HTTP 403 antes de crear la sesión.
Ver ``OriginPolicy``.

Modelo de concurrencia
----------------------
Todo lo que toca el modelo o el ``SKOPEngine`` se ejecuta en un único hilo
worker (la GPU es un recurso serial). El bucle de eventos sólo valida, enruta y
envía mensajes, así que nunca se bloquea. Las mutaciones del circuito se encolan
y el worker las aplica en una frontera de token (entre dos forwards), por lo que
un hot-swap durante una generación nunca ve un forward a medias. El circuito es
global al proceso (un modelo por servidor) y cada cambio se difunde a todas las
sesiones conectadas con un evento ``circuit``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union

import torch
from fastapi import FastAPI, WebSocket, status
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from starlette.websockets import WebSocketDisconnect, WebSocketState
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from latentflow._console import ensure_utf8_stdio
from latentflow.core.sae import FeatureError, FeatureNotFoundError, FeatureResolver, GemmaScopeResolver, ResolvedFeature
from latentflow.core.steering import DEFAULT_MODEL_ID, SKOPConfig, SKOPEngine, load_quantized_model

__all__ = [
    "DEFAULT_ALLOWED_ORIGINS", "OriginPolicy", "ServerSettings", "SettingsError", "InferenceRuntime",
    "build_runtime", "create_app",
]

logger = logging.getLogger("latentflow.api")

Event = Dict[str, Any]
TelemetryMode = Literal["none", "summary", "risky", "all"]


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
class SettingsError(ValueError):
    """Variable ``LATENTFLOW_*`` con un valor no válido (mensaje para el usuario)."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    flag = value.strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    raise SettingsError(f"{name}={value!r} no es un booleano (usa 1/0, true/false, yes/no u on/off)")


def _env_number(name: str, default: Any, kind: Callable[[str], Any], minimum: Optional[float] = None) -> Any:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        number = kind(value.strip())
    except ValueError:
        raise SettingsError(f"{name}={value!r} no es un número {'entero' if kind is int else 'real'}") from None
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise SettingsError(f"{name}={value!r} fuera de rango" + (f" (mínimo {minimum})" if minimum is not None else ""))
    return number


def _env_list(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


# Orígenes del frontend local. Vite está fijado a estos puertos (`strictPort`),
# así que nunca salta a otro que quede fuera de la lista.
DEFAULT_ALLOWED_ORIGINS: Tuple[str, ...] = (
    "http://localhost:5173",      # vite dev
    "http://127.0.0.1:5173",
    "http://localhost:4173",      # vite preview
    "http://127.0.0.1:4173",
    "tauri://localhost",          # Tauri (macOS / Linux)
    "http://tauri.localhost",     # Tauri v2 (Windows)
    "https://tauri.localhost",    # Tauri v1 (Windows)
)


@dataclass
class ServerSettings:
    model_id: str = DEFAULT_MODEL_ID
    load_in_4bit: bool = True
    calibration_path: Optional[str] = None
    alpha: float = 1.0
    steering_on_start: bool = True
    max_prompt_tokens: int = 4096
    # Protección cross-site del WebSocket (ver `OriginPolicy`).
    allowed_origins: Tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    allow_missing_origin: bool = True
    # Pesos locales de SAEs (Gemma Scope); además se consulta la caché de Hugging Face.
    sae_dir: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ServerSettings":
        """Lee ``LATENTFLOW_*``; un valor mal escrito lanza ``SettingsError`` con su nombre."""
        return cls(
            model_id=os.environ.get("LATENTFLOW_MODEL_ID", DEFAULT_MODEL_ID),
            load_in_4bit=_env_bool("LATENTFLOW_LOAD_IN_4BIT", True),
            calibration_path=os.environ.get("LATENTFLOW_CALIBRATION") or None,
            alpha=_env_number("LATENTFLOW_ALPHA", 1.0, float),
            steering_on_start=_env_bool("LATENTFLOW_STEERING_ON_START", True),
            max_prompt_tokens=_env_number("LATENTFLOW_MAX_PROMPT_TOKENS", 4096, int, minimum=1),
            allowed_origins=_env_list("LATENTFLOW_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS),
            allow_missing_origin=_env_bool("LATENTFLOW_ALLOW_MISSING_ORIGIN", True),
            sae_dir=os.environ.get("LATENTFLOW_SAE_DIR") or None,
        )


def normalize_origin(origin: str) -> str:
    """Esquema y host no distinguen mayúsculas; se ignora una barra final."""
    return origin.strip().rstrip("/").lower()


class OriginPolicy:
    """Validación del header ``Origin`` en el handshake del WebSocket.

    Los navegadores no aplican CORS a los WebSockets y el ``CORSMiddleware`` de
    Starlette sólo actúa sobre peticiones HTTP: sin esta comprobación, cualquier
    web abierta en el navegador del usuario podría conectarse a
    ``ws://127.0.0.1:8000`` y controlar el modelo (Cross-Site WebSocket Hijacking).

    - Origen en la lista blanca (comparación exacta, sin comodines de puerto): se acepta.
    - ``Origin: null`` (iframes sandbox, ``file://``, redirecciones) o varios headers
      ``Origin``: se rechaza.
    - Sin ``Origin``: los navegadores siempre lo envían en el handshake, así que su
      ausencia indica un cliente no-navegador (scripts, CLI, tests). Se acepta salvo
      ``allow_missing_origin=False``.
    - ``"*"`` en la lista desactiva la protección (sólo para depuración).
    """

    def __init__(self, allowed_origins: Iterable[str], allow_missing: bool = True) -> None:
        self.allowed = frozenset(normalize_origin(o) for o in allowed_origins)
        self.allow_missing = allow_missing
        if "*" in self.allowed:
            logger.warning("LATENTFLOW_ALLOWED_ORIGINS contiene '*': protección de Origin DESACTIVADA")

    def is_allowed(self, origins: Sequence[str]) -> bool:
        if not origins:
            return self.allow_missing
        if len(origins) > 1:
            return False
        origin = normalize_origin(origins[0])
        if origin == "null":
            return False
        return "*" in self.allowed or origin in self.allowed


# --------------------------------------------------------------------------- #
# Mensajes del cliente
# --------------------------------------------------------------------------- #
class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerateRequest(_Message):
    action: Literal["generate"]
    prompt: str = Field(min_length=1)
    request_id: Optional[str] = Field(None, max_length=128)
    max_new_tokens: int = Field(128, ge=1, le=4096)
    temperature: float = Field(0.0, ge=0.0, le=5.0, allow_inf_nan=False)  # 0 -> greedy
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    chat: bool = True
    telemetry: TelemetryMode = "risky"


class FeatureSpec(_Message):
    """Componente de la dirección de una capa: feature de SAE o, con ``None``, el r_q calibrado."""

    feature_id: Optional[str] = Field(None, min_length=1, max_length=256)
    alpha: float = Field(ge=-1e3, le=1e3, allow_inf_nan=False)


class LayerSpec(_Message):
    """Inyección en una capa: alpha (ganancia), modo y, opcionalmente, sus features."""

    layer: int = Field(ge=0)
    alpha: float = Field(ge=-1e3, le=1e3, allow_inf_nan=False)
    mode: Literal["additive", "orthogonal"]
    features: Optional[List[FeatureSpec]] = Field(None, max_length=64)


class UpdateRequest(_Message):
    action: Literal["update"]
    alpha: Optional[float] = Field(None, ge=-1e3, le=1e3, allow_inf_nan=False)
    steering: Optional[bool] = None
    skop_filter: Optional[bool] = None
    gamma_energy: Optional[float] = Field(None, gt=0.0, le=1.0)
    risk_percentile: Optional[float] = Field(None, ge=0.0, le=1.0)
    # Plan por capa: sólo estas capas reciben steering, cada una con su alpha y
    # modo; sustituye a alpha/skop_filter globales. `reset_layers` vuelve al modo global.
    layers: Optional[List[LayerSpec]] = Field(None, max_length=1024)
    reset_layers: Optional[bool] = None

    @model_validator(mode="after")
    def _check_layers(self) -> "UpdateRequest":
        if self.layers is not None:
            ids = [s.layer for s in self.layers]
            if len(ids) != len(set(ids)):
                raise ValueError("`layers` contiene capas repetidas")
            if self.reset_layers:
                raise ValueError("`layers` y `reset_layers` son excluyentes")
        return self

    def changes(self) -> Dict[str, Any]:
        changes = self.model_dump(exclude={"action"}, exclude_none=True)
        if changes.get("reset_layers") is False:
            del changes["reset_layers"]
        return changes


class CancelRequest(_Message):
    action: Literal["cancel"]


class CalibrateRequest(_Message):
    """``utility`` estima Σ_ΔK (Focus/Tail); ``positive``/``negative``, ``r_q``."""

    action: Literal["calibrate"]
    utility: List[str] = Field(min_length=1)
    positive: List[str] = Field(min_length=1)
    negative: List[str] = Field(min_length=1)


class StatusRequest(_Message):
    action: Literal["status"]


class PingRequest(_Message):
    action: Literal["ping"]


ClientMessage = Annotated[
    Union[GenerateRequest, UpdateRequest, CancelRequest, CalibrateRequest, StatusRequest, PingRequest],
    Field(discriminator="action"),
]
_client_message = TypeAdapter(ClientMessage)


def _error(code: str, message: str, **extra: Any) -> Event:
    return {"type": "error", "code": code, "message": message, **extra}


def _message_context(raw: str) -> Dict[str, str]:
    """``action`` y ``request_id`` de un mensaje que no pasó la validación, si se leen."""
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: data[k] for k in ("action", "request_id") if isinstance(data.get(k), str) and len(data[k]) <= 64}


# --------------------------------------------------------------------------- #
# Circuito (estado de steering con hot-swap en frontera de token)
# --------------------------------------------------------------------------- #
class CircuitController:
    """Cola de mutaciones del circuito. ``apply_pending`` sólo se llama desde el worker."""

    def __init__(self, engine: SKOPEngine) -> None:
        self.engine = engine
        self.version = 0
        self._lock = threading.Lock()
        self._pending: Dict[str, Any] = {}
        self._features: Dict[str, ResolvedFeature] = {}     # decoders ya resueltos (residual)
        self._layer_features: Dict[int, List[Dict[str, Any]]] = {}

    def request(self, changes: Dict[str, Any], features: Optional[Dict[str, ResolvedFeature]] = None) -> None:
        """Encola cambios; si aún no se aplicaron los anteriores, se fusionan y la
        petición más reciente prevalece (también entre `layers` y `reset_layers`,
        que se excluyen: si no, un reset posterior quedaría deshecho al aplicar)."""
        with self._lock:
            self._features.update(features or {})
            if "layers" in changes:
                self._pending.pop("reset_layers", None)
            if changes.get("reset_layers"):
                self._pending.pop("layers", None)
            self._pending.update(changes)

    def _layer_directions(self, specs: List[Dict[str, Any]]):
        """Direcciones (H, d) de las capas con features. Se calcula antes de mutar nada."""
        e = self.engine
        directions: Dict[int, torch.Tensor] = {}
        described: Dict[int, List[Dict[str, Any]]] = {}
        for spec in specs:
            feats = spec.get("features") or []
            if not feats:
                continue
            layer = spec["layer"]
            total: Optional[torch.Tensor] = None
            desc: List[Dict[str, Any]] = []
            for f in feats:
                fid = f.get("feature_id")
                if fid:
                    resolved = self._features[fid]
                    direction = e.residual_to_query(layer, resolved.vector)
                    desc.append({"feature_id": fid, "alpha": f["alpha"], "source": resolved.source,
                                 "sae_layer": resolved.layer})
                else:
                    direction = e.steering_vectors[layer]
                    desc.append({"feature_id": None, "alpha": f["alpha"], "source": "calibrated"})
                term = f["alpha"] * direction
                total = term if total is None else total + term
            directions[layer] = total
            described[layer] = desc
        return directions, described

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def apply_pending(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            changes, self._pending = self._pending, {}
        if not changes:
            return None
        e = self.engine
        checkpoint, layer_features = e.circuit_state(), self._layer_features
        try:
            self._apply(changes)
        except Exception:
            # Transaccional: un fallo a mitad no deja el circuito medio aplicado.
            e.restore_circuit_state(checkpoint)
            self._layer_features = layer_features
            raise
        self.version += 1
        return changes

    def _apply(self, changes: Dict[str, Any]) -> None:
        e = self.engine
        if "layers" in changes:
            directions, described = self._layer_directions(changes["layers"])
        refit = False
        if "gamma_energy" in changes:
            e.config.gamma_energy = changes["gamma_energy"]
            refit = True
        if "risk_percentile" in changes:
            e.config.risk_percentile = changes["risk_percentile"]
            refit = True
        if refit:
            e.fit()
        if "skop_filter" in changes:
            e.use_projection = changes["skop_filter"]
        if "alpha" in changes:
            e.alpha = changes["alpha"]
        if changes.get("reset_layers"):
            e.set_layer_directions(None)
            e.set_layer_plan(None)
            self._layer_features = {}
        if "layers" in changes:
            e.set_layer_directions(directions or None)
            e.set_layer_plan({s["layer"]: (s["alpha"], s["mode"] == "orthogonal") for s in changes["layers"]})
            self._layer_features = described
        if "steering" in changes:
            if changes["steering"]:
                e.attach()
            else:
                e.detach()

    def snapshot(self) -> Dict[str, Any]:
        e = self.engine
        risky = {l: m.nonzero().flatten().tolist() for l, m in e.risk_mask.items()} if e.is_fitted else {}
        return {
            "version": self.version,
            "calibrated": e.is_fitted,
            "steering": e.is_attached,
            "skop_filter": e.use_projection,
            "alpha": e.alpha,
            "gamma_energy": e.config.gamma_energy,
            "risk_percentile": e.config.risk_percentile,
            "normalize_rayleigh": e.config.normalize_rayleigh,
            "risk_threshold": e.risk_threshold,
            "num_risky_heads": sum(len(v) for v in risky.values()),
            "risky_heads": {str(l): v for l, v in risky.items() if v},
            "active_layers": e.active_layers,
            "layer_plan": None if e.layer_plan is None else [
                {"layer": l, "alpha": a, "mode": "orthogonal" if p else "additive",
                 **({"features": self._layer_features[l]} if l in self._layer_features else {})}
                for l, (a, p) in sorted(e.layer_plan.items())
            ],
        }


# --------------------------------------------------------------------------- #
# Telemetría
# --------------------------------------------------------------------------- #
def _r(x: float) -> float:
    return round(float(x), 6)


def build_telemetry(
    engine: SKOPEngine,
    live: Dict[int, torch.Tensor],
    mode: TelemetryMode,
    unsteered: Optional[Dict[int, torch.Tensor]] = None,
) -> Optional[Event]:
    """Resumen del cociente de Rayleigh en vivo (batch 0) para el token actual.

    ``rayleigh`` es R(q) de la query con steering y ``unsteered`` R(q − δ) de la misma
    query sin el vector inyectado: son la misma magnitud y se comparan entre sí.
    ``baseline`` (R de la dirección de steering) y ``risk_threshold`` son la escala
    del ranking de riesgo, no la de las queries.
    """
    if mode == "none":
        return None
    if not live:
        return {"available": False}
    unsteered = unsteered or live
    layers = sorted(live)
    scores = torch.stack([live[l][0] for l in layers])  # (L, H)
    base = torch.stack([unsteered[l][0] for l in layers])
    num_heads = scores.shape[1]
    layer_pos, head = divmod(int(scores.argmax()), num_heads)
    threshold = engine.risk_threshold
    telemetry: Event = {
        "available": True,
        "summary": {
            "mean": _r(scores.mean()),
            "max": _r(scores.max()),
            "mean_unsteered": _r(base.mean()),
            "argmax": {"layer": layers[layer_pos], "head": head},
            "risk_threshold": None if threshold is None else _r(threshold),
        },
    }
    if mode in ("risky", "all"):
        heads = []
        active = set(engine.active_layers) if engine.is_attached else set()
        for i, l in enumerate(layers):
            risky = engine.risk_mask[l]
            projected_layer = l in active and engine.projection_for(l)
            for h in range(num_heads):
                if mode == "risky" and not risky[h]:
                    continue
                heads.append(
                    {
                        "layer": l,
                        "head": h,
                        "rayleigh": _r(scores[i, h]),
                        "unsteered": _r(base[i, h]),
                        "baseline": _r(engine.rayleigh[l][h]),
                        "risky": bool(risky[h]),
                        "steered": l in active,
                        "projected": bool(risky[h]) and projected_layer,
                    }
                )
        telemetry["heads"] = heads
    return telemetry


# --------------------------------------------------------------------------- #
# Generación (se ejecuta en el hilo worker)
# --------------------------------------------------------------------------- #
class _CancelCriteria(StoppingCriteria):
    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


class _TokenStreamer(BaseStreamer):
    """Recibe cada token de ``generate``; el primer ``put`` es el prompt."""

    def __init__(self, on_token: Callable[[int], None]) -> None:
        self.on_token = on_token
        self._prompt_seen = False

    def put(self, value: torch.Tensor) -> None:
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        self.on_token(int(value.reshape(-1)[0]))

    def end(self) -> None:
        pass


class GenerationJob:
    def __init__(
        self,
        runtime: "InferenceRuntime",
        request: GenerateRequest,
        request_id: str,
        emit: Callable[[Event], None],
    ) -> None:
        self.runtime = runtime
        self.request = request
        self.request_id = request_id
        self.emit = emit
        self.cancel_event = threading.Event()
        self.finished = threading.Event()
        self.token_ids: List[int] = []
        self._text = ""

    def run(self) -> None:
        rt, req = self.runtime, self.request
        engine = rt.engine
        started = time.perf_counter()
        reason = "error"
        try:
            rt.apply_circuit(applied_at_index=None, request_id=self.request_id)
            if self.cancel_event.is_set():
                reason = "cancelled"
                return
            input_ids, attention_mask = rt.encode_prompt(req.prompt, req.chat)
            if input_ids.shape[-1] > rt.settings.max_prompt_tokens:
                self.emit(_error("prompt_too_long", f"El prompt supera {rt.settings.max_prompt_tokens} tokens",
                                 request_id=self.request_id))
                return
            self.emit({
                "type": "start",
                "request_id": self.request_id,
                "prompt_tokens": int(input_ids.shape[-1]),
                "circuit": rt.circuit.snapshot(),
            })
            probing = req.telemetry != "none" and engine.is_fitted
            if probing:
                engine.attach_telemetry()
            try:
                rt.model.generate(**self._generate_kwargs(input_ids, attention_mask))
            finally:
                if probing:
                    engine.detach_telemetry()
            reason = self._end_reason()
        except Exception as exc:  # el worker nunca debe morir por una petición
            logger.exception("Fallo en la generación %s", self.request_id)
            self.emit(_error("generation_failed", f"{type(exc).__name__}: {exc}", request_id=self.request_id))
        finally:
            self.finished.set()
            elapsed = time.perf_counter() - started
            n = len(self.token_ids)
            self.emit({
                "type": "end",
                "request_id": self.request_id,
                "reason": reason,
                "num_tokens": n,
                "text": self._text,
                "elapsed_s": round(elapsed, 4),
                "tokens_per_s": round(n / elapsed, 2) if elapsed > 0 else None,
            })

    def _generate_kwargs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        req, tok = self.request, self.runtime.tokenizer
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": req.max_new_tokens,
            "streamer": _TokenStreamer(self._on_token),
            "stopping_criteria": StoppingCriteriaList([_CancelCriteria(self.cancel_event)]),
            "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        }
        if req.temperature > 0:
            kwargs.update(do_sample=True, temperature=req.temperature, top_p=req.top_p)
        else:
            kwargs.update(do_sample=False)
        return kwargs

    def _on_token(self, token_id: int) -> None:
        """Frontera de token: emite el token con su telemetría y aplica hot-swaps."""
        rt = self.runtime
        index = len(self.token_ids)
        self.token_ids.append(token_id)
        # Telemetría del forward que acaba de producir este token.
        live, unsteered = rt.engine.collect_telemetry_pair() if rt.engine.telemetry_attached else ({}, {})
        self.emit({
            "type": "token",
            "request_id": self.request_id,
            "index": index,
            "token": self._decode_delta(),
            "token_id": token_id,
            "circuit_version": rt.circuit.version,
            "telemetry": build_telemetry(rt.engine, live, self.request.telemetry, unsteered),
        })
        rt.apply_circuit(applied_at_index=index, request_id=self.request_id)

    def _decode_delta(self) -> str:
        # Decodificación incremental: los tokenizers SentencePiece/BPE no son
        # decodificables token a token (espacios, bytes UTF-8 partidos).
        text = self.runtime.tokenizer.decode(self.token_ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""  # carácter multibyte incompleto: se emitirá con el siguiente token
        delta, self._text = text[len(self._text):], text
        return delta

    def _end_reason(self) -> str:
        if self.cancel_event.is_set():
            return "cancelled"
        if self.token_ids and self.token_ids[-1] in self.runtime.eos_token_ids:
            return "eos"
        if len(self.token_ids) >= self.request.max_new_tokens:
            return "length"
        return "stop"


# --------------------------------------------------------------------------- #
# Runtime compartido
# --------------------------------------------------------------------------- #
class InferenceRuntime:
    """Modelo + tokenizer + SKOPEngine, servidos por un único hilo worker."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        engine: SKOPEngine,
        settings: ServerSettings,
        feature_resolver: Optional[FeatureResolver] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.settings = settings
        self.feature_resolver: FeatureResolver = feature_resolver or GemmaScopeResolver(settings.sae_dir)
        self.circuit = CircuitController(engine)
        self.sessions: Set["StreamSession"] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="latentflow-gpu")
        self._inflight = 0
        self.eos_token_ids = self._collect_eos_ids()

    def _collect_eos_ids(self) -> Set[int]:
        ids: Set[int] = set()
        gen_cfg = getattr(self.model, "generation_config", None)
        for value in (getattr(gen_cfg, "eos_token_id", None), getattr(self.tokenizer, "eos_token_id", None)):
            if isinstance(value, int):
                ids.add(value)
            elif value is not None:
                ids.update(int(v) for v in value)
        return ids

    # ------------------------------------------------------------ ciclo de vida
    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        if self.engine.is_fitted and self.settings.steering_on_start:
            self.circuit.request({"steering": True})
            await self.submit(self.apply_circuit)

    async def shutdown(self) -> None:
        for session in list(self.sessions):
            session.cancel_active()
        await asyncio.to_thread(self._executor.shutdown, True, cancel_futures=True)
        self.engine.detach()
        self.engine.detach_telemetry()

    @property
    def busy(self) -> bool:
        return self._inflight > 0

    def submit(self, fn: Callable[..., Any], *args: Any) -> "asyncio.Future[Any]":
        """Encola trabajo en el worker. Llamar sólo desde el bucle de eventos."""
        assert self.loop is not None, "runtime no iniciado"
        self._inflight += 1
        future = self.loop.run_in_executor(self._executor, fn, *args)

        def _done(f: "asyncio.Future[Any]") -> None:
            self._inflight -= 1
            if not f.cancelled() and f.exception() is not None:
                logger.error("Trabajo del worker falló", exc_info=f.exception())

        future.add_done_callback(_done)
        return future

    # --------------------------------------------------- utilidades del worker
    def resolve_features(self, feature_ids: List[str]) -> Dict[str, ResolvedFeature]:
        """Lee los decoders (E/S, sin tocar el modelo): se ejecuta fuera del worker."""
        d_model = self.engine.hidden_size
        return {fid: self.feature_resolver.resolve(fid, d_model) for fid in feature_ids}

    def apply_circuit(self, applied_at_index: Optional[int] = None, request_id: Optional[str] = None) -> None:
        try:
            changes = self.circuit.apply_pending()
        except Exception as exc:  # nunca debe tumbar una generación en curso
            logger.exception("No se pudo aplicar el circuito")
            self.broadcast(_error("circuit_apply_failed", f"{type(exc).__name__}: {exc}", request_id=request_id))
            return
        if changes is not None:
            self.broadcast({
                "type": "circuit",
                "changes": changes,
                "applied_at_index": applied_at_index,
                "request_id": request_id,
                "circuit": self.circuit.snapshot(),
            })

    def encode_prompt(self, prompt: str, chat: bool):
        tok = self.tokenizer
        if chat and getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            enc = tok(text, return_tensors="pt", add_special_tokens=False)
        else:
            enc = tok(prompt, return_tensors="pt")
        device = self.engine._input_device()
        input_ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask")
        mask = torch.ones_like(input_ids) if mask is None else mask.to(device)
        return input_ids, mask

    def calibrate(self, utility: List[str], positive: List[str], negative: List[str]) -> Dict[str, Any]:
        was_attached = self.engine.is_attached
        self.engine.detach()
        try:
            self.engine.calibrate(utility, positive, negative).fit()
        finally:
            if was_attached and self.engine.is_fitted:
                self.engine.attach()
        self.circuit.version += 1
        return self.circuit.snapshot()

    # -------------------------------------------------------------- difusión
    def broadcast(self, event: Event) -> None:
        """Thread-safe: envía el evento a todas las sesiones conectadas."""
        self.call_soon(self._broadcast_now, event)

    def _broadcast_now(self, event: Event) -> None:
        for session in list(self.sessions):
            session.send(event)

    def call_soon(self, fn: Callable[..., Any], *args: Any) -> None:
        if self.loop is None:
            return
        try:
            self.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:  # bucle cerrado durante el apagado
            pass


def build_runtime(settings: ServerSettings) -> InferenceRuntime:
    """Carga el modelo cuantizado (bloqueante; se llama desde un hilo)."""
    logger.info("Cargando %s (4bit=%s)", settings.model_id, settings.load_in_4bit)
    model, tokenizer = load_quantized_model(settings.model_id, load_in_4bit=settings.load_in_4bit)
    engine = SKOPEngine(model, SKOPConfig(alpha=settings.alpha), tokenizer=tokenizer)
    if settings.calibration_path:
        logger.info("Cargando calibración SKOP de %s", settings.calibration_path)
        engine.load(settings.calibration_path)
    return InferenceRuntime(model, tokenizer, engine, settings, GemmaScopeResolver(settings.sae_dir))


# --------------------------------------------------------------------------- #
# Sesión WebSocket
# --------------------------------------------------------------------------- #
@dataclass
class _ActiveGeneration:
    job: GenerationJob
    # None mientras la generación espera a los `update` anteriores (barrera de sesión).
    future: Optional["asyncio.Future[Any]"] = field(default=None, repr=False)


class StreamSession:
    def __init__(self, websocket: WebSocket, runtime: InferenceRuntime) -> None:
        self.ws = websocket
        self.runtime = runtime
        self.outbox: "asyncio.Queue[Event]" = asyncio.Queue()
        self.active: Optional[_ActiveGeneration] = None
        # Barrera de la sesión: los `update` se aplican en orden aunque alguno espere a
        # leer un SAE, y un `generate` no llega al worker hasta que terminan los
        # `update` recibidos antes que él (el lock de asyncio despierta en orden FIFO
        # y las tareas arrancan en el orden en que se crean).
        self._update_lock = asyncio.Lock()
        self._tasks: Set["asyncio.Task[None]"] = set()

    # ------------------------------------------------------------- envío
    def send(self, event: Event) -> None:
        """Encola un evento (sólo desde el bucle de eventos)."""
        self.outbox.put_nowait(event)

    def send_threadsafe(self, event: Event) -> None:
        self.runtime.call_soon(self.outbox.put_nowait, event)

    async def _sender(self) -> None:
        while True:
            event = await self.outbox.get()
            try:
                payload = json.dumps(event, allow_nan=False)
            except (TypeError, ValueError) as exc:
                payload = json.dumps(_error("serialization_failed", str(exc)))
            await self.ws.send_text(payload)

    # ------------------------------------------------------------- bucle
    async def run(self) -> None:
        await self.ws.accept()
        self.runtime.sessions.add(self)
        sender = asyncio.create_task(self._sender())
        try:
            self.send({
                "type": "ready",
                "model": self.runtime.settings.model_id,
                "geometry": vars(self.runtime.engine.geometry),
                "layers": self.runtime.engine.layer_ids,
                "hidden_size": self.runtime.engine.hidden_size,
                "sae": self.runtime.feature_resolver.describe(),
                "circuit": self.runtime.circuit.snapshot(),
            })
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                raw = message.get("text")
                if raw is None:
                    raw = (message.get("bytes") or b"").decode("utf-8", errors="replace")
                self._dispatch(raw)
                if sender.done():  # el socket murió al enviar
                    break
        except WebSocketDisconnect:
            pass
        finally:
            self.runtime.sessions.discard(self)
            self.cancel_active()
            for task in list(self._tasks):
                task.cancel()
            await self._drain_and_stop(sender)

    async def _drain_and_stop(self, sender: "asyncio.Task[None]") -> None:
        if self.ws.application_state == WebSocketState.CONNECTED and not sender.done():
            try:
                await asyncio.wait_for(self._wait_outbox_empty(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        sender.cancel()
        try:
            await sender
        except (asyncio.CancelledError, Exception):
            pass

    async def _wait_outbox_empty(self) -> None:
        while not self.outbox.empty():
            await asyncio.sleep(0.005)

    def cancel_active(self) -> None:
        if self.active is not None:
            self.active.job.cancel_event.set()

    # ------------------------------------------------------------- acciones
    def _dispatch(self, raw: str) -> None:
        try:
            msg = _client_message.validate_json(raw)
        except ValidationError as exc:
            details = json.loads(exc.json(include_url=False))
            first = details[0] if details else None
            where = ".".join(str(p) for p in first.get("loc", ())) if first else ""
            summary = f"Mensaje no válido ({where}: {first['msg']})" if first and where else "Mensaje no válido"
            # `action` y `request_id` permiten al cliente saber qué petición falló
            # (p. ej. liberar el chat si era un `generate`).
            self.send(_error("invalid_message", summary, details=details, **_message_context(raw)))
            return
        handler = {
            GenerateRequest: self._on_generate,
            UpdateRequest: self._on_update,
            CancelRequest: self._on_cancel,
            CalibrateRequest: self._on_calibrate,
            StatusRequest: self._on_status,
            PingRequest: self._on_ping,
        }[type(msg)]
        handler(msg)

    def _on_generate(self, msg: GenerateRequest) -> None:
        if self.active is not None and not self.active.job.finished.is_set():
            self.send(_error("busy", "Ya hay una generación activa en esta sesión",
                             request_id=self.active.job.request_id))
            return
        request_id = msg.request_id or uuid.uuid4().hex[:12]
        job = GenerationJob(self.runtime, msg, request_id, self.send_threadsafe)
        active = self.active = _ActiveGeneration(job)
        waiting = self._update_lock.locked()
        self.send({"type": "accepted", "request_id": request_id, "queued": self.runtime.busy or waiting})
        self._spawn(self._start_generation(active))

    async def _start_generation(self, active: _ActiveGeneration) -> None:
        # El prefill debe ver el circuito con las features ya resueltas: espera a que
        # los `update` anteriores terminen. Una cancelación durante la espera llega
        # al job, que termina con `reason: "cancelled"` sin hacer forwards.
        async with self._update_lock:
            active.future = self.runtime.submit(active.job.run)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_update(self, msg: UpdateRequest) -> None:
        changes = msg.changes()
        if not changes:
            self.send(_error("empty_update", "La actualización no contiene cambios"))
            return
        if not self.runtime.engine.is_fitted:
            self.send(_error("not_calibrated", "El circuito no está calibrado; usa 'calibrate' primero"))
            return
        unknown = sorted({s["layer"] for s in changes.get("layers", [])} - set(self.runtime.engine.layer_ids))
        if unknown:
            self.send(_error("invalid_layer", f"Capas fuera del rango calibrado: {unknown}",
                             valid_layers=self.runtime.engine.layer_ids))
            return
        self._spawn(self._apply_update(changes))

    async def _apply_update(self, changes: Dict[str, Any]) -> None:
        async with self._update_lock:
            feature_ids = sorted({
                f["feature_id"] for s in changes.get("layers", []) for f in (s.get("features") or [])
                if f.get("feature_id")
            })
            resolved: Dict[str, ResolvedFeature] = {}
            if feature_ids:
                try:
                    resolved = await asyncio.to_thread(self.runtime.resolve_features, feature_ids)
                except FeatureError as exc:
                    code = "feature_not_found" if isinstance(exc, FeatureNotFoundError) else "feature_invalid"
                    self.send(_error(code, str(exc)))
                    return
            deferred = self.runtime.busy
            self.runtime.circuit.request(changes, resolved)
            # Si hay una generación en curso la aplicará en la siguiente frontera de
            # token; este trabajo encolado la recoge si no queda ninguna.
            self.runtime.submit(self.runtime.apply_circuit)
            self.send({
                "type": "update_accepted",
                "changes": changes,
                "deferred": deferred,
                **({"features": [{"feature_id": f.feature_id, "source": f.source, "sae_layer": f.layer}
                                 for f in resolved.values()]} if resolved else {}),
            })

    def _on_cancel(self, msg: CancelRequest) -> None:
        if self.active is None or self.active.job.finished.is_set():
            self.send(_error("idle", "No hay ninguna generación activa"))
            return
        self.active.job.cancel_event.set()

    def _on_calibrate(self, msg: CalibrateRequest) -> None:
        if len(msg.positive) != len(msg.negative):
            self.send(_error("invalid_message", "positive y negative deben tener la misma longitud", action="calibrate"))
            return
        if self.runtime.tokenizer is None:
            self.send(_error("no_tokenizer", "El runtime no tiene tokenizer"))
            return
        future = self.runtime.submit(self.runtime.calibrate, msg.utility, msg.positive, msg.negative)

        def _done(f: "asyncio.Future[Any]") -> None:
            if f.cancelled():
                return
            if f.exception() is not None:
                self.send(_error("calibration_failed", f"{type(f.exception()).__name__}: {f.exception()}"))
                return
            snapshot = f.result()
            self.send({"type": "calibrated", "num_pairs": len(msg.positive), "num_utility": len(msg.utility),
                       "circuit": snapshot})
            self.runtime._broadcast_now({"type": "circuit", "changes": {"calibrated": True},
                                         "applied_at_index": None, "request_id": None, "circuit": snapshot})

        future.add_done_callback(_done)

    def _on_status(self, msg: StatusRequest) -> None:
        active = self.active is not None and not self.active.job.finished.is_set()
        self.send({
            "type": "status",
            "circuit": self.runtime.circuit.snapshot(),
            "busy": self.runtime.busy,
            "active_request_id": self.active.job.request_id if active else None,
            "sessions": len(self.runtime.sessions),
        })

    def _on_ping(self, msg: PingRequest) -> None:
        self.send({"type": "pong", "server_time": time.time()})


# --------------------------------------------------------------------------- #
# Aplicación
# --------------------------------------------------------------------------- #
def create_app(
    settings: Optional[ServerSettings] = None,
    runtime_factory: Callable[[ServerSettings], InferenceRuntime] = build_runtime,
) -> FastAPI:
    settings = settings or ServerSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = await asyncio.to_thread(runtime_factory, settings)
        await runtime.start()
        app.state.runtime = runtime
        try:
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(title="BrainSteer / LatentFlow", lifespan=lifespan)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        runtime: InferenceRuntime = app.state.runtime
        return {
            "status": "ok",
            "model": settings.model_id,
            "busy": runtime.busy,
            "sessions": len(runtime.sessions),
            "circuit": runtime.circuit.snapshot(),
        }

    origin_policy = OriginPolicy(settings.allowed_origins, settings.allow_missing_origin)

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        origins = websocket.headers.getlist("origin")
        if not origin_policy.is_allowed(origins):
            # Cerrar antes de `accept()` hace que el servidor ASGI responda HTTP 403
            # al handshake: la sesión nunca se crea y no se envía ningún dato.
            client = websocket.client.host if websocket.client else "?"
            logger.warning("WebSocket rechazado: Origin %s no permitido (cliente %s)", origins or "<ausente>", client)
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="origin not allowed")
            return
        await StreamSession(websocket, app.state.runtime).run()

    return app


def __getattr__(name: str) -> Any:
    # `uvicorn latentflow.api.server:app` sigue funcionando, pero la app (y la lectura
    # de LATENTFLOW_*) se crea al pedirla, no al importar el módulo.
    if name == "app":
        globals()["app"] = created = create_app()
        return created
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> int:
    """CLI ``latentflow-server``: valida la configuración antes de cargar nada."""
    import uvicorn

    ensure_utf8_stdio()
    try:
        settings = ServerSettings.from_env()
        port = _env_number("LATENTFLOW_PORT", 8000, int, minimum=1)
        if port > 65535:
            raise SettingsError(f"LATENTFLOW_PORT={port} fuera de rango (1–65535)")
    except SettingsError as exc:
        print(f"latentflow-server: error: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(settings), host=os.environ.get("LATENTFLOW_HOST", "127.0.0.1"), port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
