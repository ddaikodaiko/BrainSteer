"""
latentflow.tools.initializer
============================

CLI ``latentflow-calibrate``: genera el artefacto de calibración SKOP a partir de
datasets **locales** y lo guarda en el formato de ``SKOPEngine.save()`` (Σ_ΔK y
``r_q`` en float64), listo para que el servidor lo cargue en frío con
``LATENTFLOW_CALIBRATION``. No descarga nada: los datos se pasan como ficheros.

Uso::

    latentflow-calibrate --utility data/utility.jsonl --concept data/concept.jsonl \\
        -o calib/gemma2_skop.pt                         # Gemma-2-2B-IT en 4 bit (GPU)
    latentflow-calibrate ... --quant none               # sin bitsandbytes (CPU)
    latentflow-calibrate ... --dev                      # modelo tiny offline (pruebas)

Formatos (``.jsonl``: un registro por línea; ``.json``: una lista de registros):

- Utility set (Σ_ΔK, Focus/Tail): ``{"text": ...}``, estilo Alpaca
  ``{"instruction", "input"?, "output"?}``, estilo GSM8K ``{"question", "answer"?}``
  o una cadena.
- Concept set (``r_q``): ``{"positive": ..., "negative": ...}``.

Para preparar los datos en una máquina con acceso a Hugging Face::

    from datasets import load_dataset
    load_dataset("tatsu-lab/alpaca", split="train[:200]").to_json("alpaca.jsonl")
    load_dataset("openai/gsm8k", "main", split="train[:200]").to_json("gsm8k.jsonl")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from latentflow.core.steering import ARTIFACT_FORMAT, DEFAULT_MODEL_ID, SKOPConfig, SKOPEngine

__all__ = [
    "CalibrationDataError",
    "load_concept_pairs",
    "load_utility_texts",
    "parse_layers",
    "read_records",
    "utility_text",
    "main",
]

DEFAULT_OUTPUT = "calib/gemma2_skop.pt"

# Caracteres de control salvo salto de línea y tabulador.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class CalibrationDataError(ValueError):
    """Error en los datos de entrada; el CLI lo muestra sin traceback."""


# --------------------------------------------------------------------------- #
# Lectura y limpieza de datos
# --------------------------------------------------------------------------- #
def read_records(path: Path) -> List[Tuple[int, Any]]:
    """``(línea o índice, registro)`` de un ``.jsonl`` o de un ``.json`` con una lista."""
    if not path.is_file():
        raise CalibrationDataError(f"{path}: no existe o no es un fichero")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CalibrationDataError(f"{path}: JSON no válido ({exc})") from None
        if not isinstance(data, list):
            raise CalibrationDataError(f"{path}: un .json debe contener una lista de registros")
        return list(enumerate(data, start=1))
    records = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append((lineno, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise CalibrationDataError(f"{path}:{lineno}: JSON no válido ({exc.msg})") from None
    return records


def clean_text(text: str, banned_tokens: Iterable[str] = ()) -> str:
    """Quita caracteres de control y literales de tokens especiales (``<pad>``, ``<eos>``...).

    Un token especial escrito en los datos se tokenizaría como tal (p. ej. un pad
    con ``attention_mask = 1``) y contaminaría la geometría de atención.
    """
    text = _CONTROL_CHARS.sub("", text)
    for token in banned_tokens:
        if token:
            text = text.replace(token, " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def utility_text(record: Any) -> str:
    """Texto de un registro del utility set (texto libre, Alpaca o GSM8K)."""
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        raise TypeError("se esperaba un objeto JSON o una cadena")
    if isinstance(record.get("text"), str):
        return record["text"]
    if isinstance(record.get("instruction"), str):
        parts = [record["instruction"], record.get("input") or "", record.get("output") or ""]
        return "\n\n".join(p for p in parts if isinstance(p, str) and p.strip())
    if isinstance(record.get("question"), str):
        answer = record.get("answer")
        return record["question"] + (f"\n\n{answer}" if isinstance(answer, str) and answer.strip() else "")
    raise TypeError("faltan los campos 'text', 'instruction' o 'question'")


def _sample(items: List[Any], limit: Optional[int], seed: int) -> List[Any]:
    """Submuestreo reproducible que conserva el orden original."""
    if limit is None or len(items) <= limit:
        return items
    keep = sorted(random.Random(seed).sample(range(len(items)), limit))
    return [items[i] for i in keep]


def load_utility_texts(
    paths: Sequence[Path], banned_tokens: Iterable[str] = (), limit: Optional[int] = None, seed: int = 0
) -> List[str]:
    banned = list(banned_tokens)
    texts: List[str] = []
    seen = set()
    for path in paths:
        for where, record in read_records(path):
            try:
                text = clean_text(utility_text(record), banned)
            except TypeError as exc:
                raise CalibrationDataError(f"{path}:{where}: {exc}") from None
            if text and text not in seen:  # sin vacíos ni duplicados
                seen.add(text)
                texts.append(text)
    if not texts:
        raise CalibrationDataError("el utility set no contiene ningún texto válido")
    return _sample(texts, limit, seed)


def load_concept_pairs(
    path: Path, banned_tokens: Iterable[str] = (), limit: Optional[int] = None, seed: int = 0
) -> Tuple[List[str], List[str]]:
    banned = list(banned_tokens)
    pairs: List[Tuple[str, str]] = []
    for where, record in read_records(path):
        if not isinstance(record, dict) or not all(isinstance(record.get(k), str) for k in ("positive", "negative")):
            raise CalibrationDataError(f"{path}:{where}: se esperaba {{\"positive\": str, \"negative\": str}}")
        pos, neg = clean_text(record["positive"], banned), clean_text(record["negative"], banned)
        if not pos or not neg:
            raise CalibrationDataError(f"{path}:{where}: 'positive' y 'negative' no pueden quedar vacíos")
        pairs.append((pos, neg))
    if not pairs:
        raise CalibrationDataError(f"{path}: el concept set está vacío")
    pairs = _sample(pairs, limit, seed)
    return [p for p, _ in pairs], [n for _, n in pairs]


def parse_layers(spec: Optional[str]) -> Optional[List[int]]:
    """``"0-3,8,10-11"`` -> ``[0, 1, 2, 3, 8, 10, 11]``; ``None``/``"all"`` = todas."""
    if spec is None or spec.strip().lower() in ("", "all"):
        return None
    layers: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise CalibrationDataError(f"--layers: '{part}' no es un índice ni un rango (p. ej. 0-25)")
        lo, hi = int(match.group(1)), int(match.group(2) or match.group(1))
        if hi < lo:
            raise CalibrationDataError(f"--layers: rango invertido '{part}'")
        layers.extend(range(lo, hi + 1))
    return sorted(set(layers))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _special_tokens(tokenizer: Any) -> List[str]:
    tokens = getattr(tokenizer, "all_special_tokens", None)
    if tokens is None:
        tokens = [getattr(tokenizer, name, None) for name in ("pad_token", "eos_token", "bos_token", "unk_token")]
    return [t for t in tokens if isinstance(t, str) and t]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latentflow-calibrate",
        description=(
            "Calibra SKOP: Σ_ΔK (Focus/Tail) sobre un utility set y r_q sobre pares de concepto, "
            "y guarda el artefacto para LATENTFLOW_CALIBRATION."
        ),
    )
    data = parser.add_argument_group("datos (locales, .jsonl o .json)")
    data.add_argument("--utility", type=Path, action="append", required=True, metavar="PATH",
                      help="utility set para Σ_ΔK; repetible para mezclar fuentes (p. ej. Alpaca y GSM8K)")
    data.add_argument("--concept", type=Path, required=True, metavar="PATH",
                      help="pares {positive, negative} para la dirección r_q")
    data.add_argument("--max-utility", type=int, default=400, help="máximo de textos de utilidad (0 = todos)")
    data.add_argument("--max-concept", type=int, default=0, help="máximo de pares de concepto (0 = todos)")
    data.add_argument("--seed", type=int, default=0, help="semilla del submuestreo")

    model = parser.add_argument_group("modelo")
    model.add_argument("--model", default=DEFAULT_MODEL_ID, help="id de Hugging Face o ruta local")
    model.add_argument("--quant", choices=("4bit", "8bit", "none"), default="4bit",
                       help="cuantización bitsandbytes (none = sin cuantizar, p. ej. en CPU)")
    model.add_argument("--dev", action="store_true", help="usa el modelo tiny de pesos aleatorios (offline)")
    model.add_argument("--dev-layers", type=int, default=26, help="capas del modelo tiny con --dev")

    skop = parser.add_argument_group("SKOP")
    defaults = SKOPConfig()
    skop.add_argument("--layers", default=None, help="capas a calibrar, p. ej. '0-25' o '6,12' (por defecto todas)")
    skop.add_argument("--tau-high", type=float, default=defaults.tau_high, help="umbral de atención del Focus Set")
    skop.add_argument("--gamma", type=float, default=defaults.gamma_energy, help="energía retenida en U_ΔK")
    skop.add_argument("--risk-percentile", type=float, default=defaults.risk_percentile)
    skop.add_argument("--token-aggregation", choices=("last", "mean"), default=defaults.token_aggregation)
    skop.add_argument("--batch-size", type=int, default=defaults.batch_size)
    skop.add_argument("--max-length", type=int, default=defaults.max_length)

    out = parser.add_argument_group("salida")
    out.add_argument("-o", "--output", type=Path, default=Path(DEFAULT_OUTPUT))
    out.add_argument("--force", action="store_true", help="sobrescribe el artefacto si ya existe")
    return parser


def _load_model(args: argparse.Namespace):
    if args.dev:
        from latentflow.tools.dev_backend import DevModelSpec, build_dev_model

        spec = DevModelSpec(num_layers=args.dev_layers, delay=0.0, calibrated=False)
        model, tokenizer = build_dev_model(spec)
        return model, tokenizer, spec.label
    from latentflow.core.steering import load_quantized_model

    model, tokenizer = load_quantized_model(
        args.model,
        load_in_4bit=args.quant == "4bit",
        quantize=args.quant != "none",
        attn_implementation="eager",  # Focus/Tail necesita los pesos de atención
    )
    return model, tokenizer, args.model


def run(args: argparse.Namespace, log=print) -> Path:
    """Ejecuta la calibración completa y devuelve la ruta del artefacto."""
    output: Path = args.output
    if output.exists() and not args.force:
        raise CalibrationDataError(f"{output} ya existe; usa --force para sobrescribirlo")
    layers = parse_layers(args.layers)
    config = SKOPConfig(
        gamma_energy=args.gamma,
        risk_percentile=args.risk_percentile,
        tau_high=args.tau_high,
        layers=layers,
        token_aggregation=args.token_aggregation,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    started = time.perf_counter()
    # Los datos se validan antes de cargar el modelo (que puede tardar minutos).
    load_utility_texts(args.utility)
    load_concept_pairs(args.concept)

    log(f"Cargando modelo {'tiny (--dev)' if args.dev else args.model} [{'dev' if args.dev else args.quant}]...")
    model, tokenizer, model_label = _load_model(args)
    engine = SKOPEngine(model, config, tokenizer=tokenizer)

    banned = _special_tokens(tokenizer)
    utility = load_utility_texts(args.utility, banned, args.max_utility or None, args.seed)
    positive, negative = load_concept_pairs(args.concept, banned, args.max_concept or None, args.seed)
    log(f"Datos: {len(utility)} textos de utilidad, {len(positive)} pares de concepto, capas {engine.layer_ids[0]}–{engine.layer_ids[-1]}")

    engine.calibrate(utility, positive, negative).fit()
    summary = engine.summary()
    metadata: Dict[str, Any] = {
        "model_id": model_label,
        "quantization": "dev" if args.dev else args.quant,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "latentflow-calibrate",
        "torch": str(torch.__version__),  # TorchVersion no es cargable con weights_only=True
        "config": {
            "tau_high": config.tau_high,
            "gamma_energy": config.gamma_energy,
            "risk_percentile": config.risk_percentile,
            "token_aggregation": config.token_aggregation,
            "max_length": config.max_length,
        },
        "datasets": {
            "utility": [{"path": str(p), "sha256": _sha256(p)} for p in args.utility],
            "concept": {"path": str(args.concept), "sha256": _sha256(args.concept)},
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    try:
        engine.save(str(tmp), metadata=metadata)
        # Verifica que el servidor podrá cargarlo (torch.load con weights_only=True).
        SKOPEngine(model, config).load(str(tmp))
        os.replace(tmp, output)  # escritura atómica: nunca queda un artefacto a medias
    finally:
        tmp.unlink(missing_ok=True)

    ranks = [r for layer in summary["layers"].values() for r in layer["rank_per_kv_head"]]
    risky = sum(len(layer["risky_heads"]) for layer in summary["layers"].values())
    log(
        f"Σ_ΔK: {engine.num_delta_pairs:,} pares Focus × Tail de {engine.num_utility_samples} secuencias · "
        f"rango medio de U_ΔK {sum(ranks) / len(ranks):.1f}/{engine.geometry.head_dim} (γ={config.gamma_energy})"
    )
    log(f"r_q: {engine.num_concept_pairs} pares · cabezas de riesgo: {risky} (τ_R={summary['risk_threshold']:.4g})")
    log(f"Artefacto {ARTIFACT_FORMAT}: {output} ({output.stat().st_size / 1e6:.2f} MB) en {time.perf_counter() - started:.1f} s")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (CalibrationDataError, ValueError, RuntimeError, OSError) as exc:
        print(f"latentflow-calibrate: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
