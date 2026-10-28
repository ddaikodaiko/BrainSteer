"""
latentflow.tools.dev_backend
============================

Backend de desarrollo sin GPU: el servidor real de LatentFlow (mismo protocolo,
worker, hooks de steering, hot-swap y telemetría) sobre un Llama diminuto con
pesos aleatorios, calibrado con pares sintéticos. Sirve para desarrollar la UI
(React Flow, chat y osciloscopio) en cualquier máquina, sin descargar modelos.

Uso::

    python -m latentflow.tools.dev_backend                 # 26 capas, ws://127.0.0.1:8000/ws/stream
    python -m latentflow.tools.dev_backend --layers 4 --delay 0
    latentflow-dev --port 8001                             # tras `pip install -e ".[server]"`

El texto generado es ruido (los pesos son aleatorios). La geometría por defecto
reproduce el número de capas y de cabezas de query de Gemma-2-2B (26 × 8); el
resto de dimensiones está reducido para que corra en CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from fastapi import FastAPI
from transformers import LlamaConfig, LlamaForCausalLM

from latentflow.api.server import InferenceRuntime, ServerSettings, create_app
from latentflow.core.steering import SKOPConfig, SKOPEngine

__all__ = ["CharTokenizer", "DevModelSpec", "build_dev_runtime", "dev_runtime_factory", "create_dev_app", "main"]

logger = logging.getLogger("latentflow.dev")


class CharTokenizer:
    """Tokenizer de caracteres sin vocabulario externo.

    Ids 0-2 especiales (pad, bos, eos); el resto mapea caracteres a
    ``3..vocab_size-1``. Implementa sólo lo que usan el servidor y el
    ``SKOPEngine``: ``__call__`` con padding a la derecha y ``decode``.
    """

    pad_token, eos_token = "<pad>", "<eos>"
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    chat_template = None

    def __init__(self, vocab_size: int = 100) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size debe ser >= 4")
        self.vocab_size = vocab_size

    def _encode(self, text: str) -> List[int]:
        return [3 + (ord(c) % (self.vocab_size - 3)) for c in text] or [3]

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        return_tensors: str = "pt",
        max_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        texts = [text] if isinstance(text, str) else list(text)
        ids = [self._encode(t)[:max_length] for t in texts]
        width = max(len(i) for i in ids)
        input_ids = torch.full((len(ids), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.long)
        for row, seq in enumerate(ids):
            input_ids[row, : len(seq)] = torch.tensor(seq)
            mask[row, : len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(ord("a") + (int(i) - 3) % 26) for i in ids if int(i) >= 3 or not skip_special_tokens)


@dataclass(frozen=True)
class DevModelSpec:
    num_layers: int = 26
    num_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 32
    hidden_size: int = 128
    intermediate_size: int = 256
    vocab_size: int = 100
    # Retardo por forward (s): simula la latencia de un modelo real para que el
    # streaming y los hot-swaps a mitad de generación sean observables en la UI.
    delay: float = 0.04
    calibrated: bool = True
    calibration_batches: int = 6
    calibration_batch_size: int = 4
    calibration_seq_len: int = 8
    seed: int = 0

    @property
    def label(self) -> str:
        return f"dev/tiny-llama-{self.num_layers}L (pesos aleatorios)"


def build_dev_runtime(settings: ServerSettings, spec: DevModelSpec = DevModelSpec()) -> InferenceRuntime:
    """Modelo tiny + tokenizer de caracteres + SKOPEngine calibrado con ruido."""
    torch.manual_seed(spec.seed)
    config = LlamaConfig(
        vocab_size=spec.vocab_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_layers,
        num_attention_heads=spec.num_heads,
        num_key_value_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
        pad_token_id=CharTokenizer.pad_token_id,
        bos_token_id=CharTokenizer.bos_token_id,
        eos_token_id=CharTokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(config).eval()
    # Sin EOS: la longitud de cada generación la decide sólo `max_new_tokens`.
    model.generation_config.eos_token_id = None
    if spec.delay > 0:
        model.register_forward_pre_hook(lambda module, args: time.sleep(spec.delay))

    tokenizer = CharTokenizer(spec.vocab_size)
    engine = SKOPEngine(model, SKOPConfig(alpha=settings.alpha), tokenizer=tokenizer)
    if spec.calibrated:
        g = torch.Generator().manual_seed(spec.seed + 1)
        shape = (spec.calibration_batch_size, spec.calibration_seq_len)

        def batch() -> Dict[str, torch.Tensor]:
            return {
                "input_ids": torch.randint(3, spec.vocab_size, shape, generator=g),
                "attention_mask": torch.ones(shape, dtype=torch.long),
            }

        for _ in range(spec.calibration_batches):
            engine.accumulate_pair_batch(batch(), batch())
        engine.finalize_calibration().fit()
    return InferenceRuntime(model, tokenizer, engine, settings)


def dev_runtime_factory(spec: DevModelSpec = DevModelSpec()) -> Callable[[ServerSettings], InferenceRuntime]:
    return lambda settings: build_dev_runtime(settings, spec)


def create_dev_app(spec: DevModelSpec = DevModelSpec(), settings: Optional[ServerSettings] = None) -> FastAPI:
    """App FastAPI real sobre el modelo tiny. Hereda de las variables de entorno
    la configuración de seguridad (``LATENTFLOW_ALLOWED_ORIGINS``...)."""
    settings = settings or ServerSettings.from_env()
    settings = dataclasses.replace(settings, model_id=spec.label, load_in_4bit=False, calibration_path=None)
    return create_app(settings, dev_runtime_factory(spec))


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    d = DevModelSpec()
    parser = argparse.ArgumentParser(
        prog="latentflow-dev",
        description="Servidor LatentFlow de desarrollo con un modelo tiny de pesos aleatorios (sin GPU).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="interfaz de escucha (por defecto sólo loopback)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--layers", type=int, default=d.num_layers)
    parser.add_argument("--heads", type=int, default=d.num_heads)
    parser.add_argument("--kv-heads", type=int, default=d.num_kv_heads)
    parser.add_argument("--head-dim", type=int, default=d.head_dim)
    parser.add_argument("--hidden", type=int, default=d.hidden_size)
    parser.add_argument("--delay", type=float, default=d.delay, help="segundos de retardo por forward")
    parser.add_argument("--uncalibrated", action="store_true", help="arranca sin calibración SKOP")
    parser.add_argument("--seed", type=int, default=d.seed)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import uvicorn

    args = _parse_args(argv)
    spec = DevModelSpec(
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        hidden_size=args.hidden,
        intermediate_size=2 * args.hidden,
        delay=args.delay,
        calibrated=not args.uncalibrated,
        seed=args.seed,
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning("Escuchando en %s: el servidor queda accesible desde la red", args.host)
    logger.info("%s en ws://%s:%d/ws/stream", spec.label, args.host, args.port)
    uvicorn.run(create_dev_app(spec), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
