"""
latentflow.tools.probe
======================

CLI ``latentflow-probe``: muestra a simple vista el efecto del steering de un
artefacto de calibración, sin servidor ni cabina. Con el mismo prompt y
decodificación greedy, genera una vez sin steering y después para cada α y cada
modo (``additive`` = steering vanilla, ``skop`` = proyección en las cabezas de
riesgo), aplicando ``r_q`` en todas las capas calibradas o en las que se indiquen.
Sirve para elegir capas y rango de α antes de evaluar en serio.

Uso::

    python -m latentflow.tools.probe --calibration calib/gemma2_skop.pt \\
        --prompt "Write a short email to my boss asking for a day off." \\
        --alphas 0,2,-2,4,-4,8,-8 --modes additive,skop -o probe.jsonl

Cierra antes el servidor: cada proceso carga su propia copia del modelo en la GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from latentflow._console import ensure_utf8_stdio
from latentflow.core.steering import DEFAULT_MODEL_ID, SKOPConfig, SKOPEngine
from latentflow.tools.initializer import CalibrationDataError, _load_model, parse_layers

__all__ = ["build_parser", "main", "parse_alphas", "run"]

MODES = ("additive", "skop")


def parse_alphas(spec: str) -> List[float]:
    """``"0,4,-4"`` -> ``[0.0, 4.0, -4.0]`` (orden y repeticiones como vienen)."""
    try:
        alphas = [float(part) for part in spec.split(",") if part.strip()]
    except ValueError:
        raise CalibrationDataError(f"--alphas: '{spec}' no es una lista de números (p. ej. 0,4,-4)") from None
    if not alphas:
        raise CalibrationDataError("--alphas está vacío")
    return alphas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latentflow-probe",
        description="Compara generaciones con y sin steering SKOP para un mismo prompt (greedy).",
    )
    parser.add_argument("--calibration", type=Path, required=True, help="artefacto de latentflow-calibrate")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--alphas", default="0,2,-2,4,-4,8,-8", help="intensidades separadas por comas")
    parser.add_argument("--modes", default="additive,skop", help="additive (vanilla), skop o ambos")
    parser.add_argument("--layers", default=None, help="capas con steering, p. ej. '4-12' (por defecto todas)")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--no-chat", action="store_true", help="no aplicar la plantilla de chat")
    parser.add_argument("-o", "--output", type=Path, default=None, help="guarda cada generación en JSONL")

    model = parser.add_argument_group("modelo")
    model.add_argument("--model", default=DEFAULT_MODEL_ID)
    model.add_argument("--quant", choices=("4bit", "8bit", "none"), default="4bit")
    model.add_argument("--dev", action="store_true", help="modelo tiny de pesos aleatorios (pruebas)")
    model.add_argument("--dev-layers", type=int, default=26)
    return parser


def _encode(tokenizer: Any, prompt: str, chat: bool, device: torch.device):
    if chat and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    else:
        enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    return input_ids, (torch.ones_like(input_ids) if mask is None else mask.to(device))


@torch.no_grad()
def _generate(engine: SKOPEngine, tokenizer: Any, prompt: str, chat: bool, max_new_tokens: int) -> Dict[str, Any]:
    input_ids, mask = _encode(tokenizer, prompt, chat, engine._input_device())
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    started = time.perf_counter()
    out = engine.model.generate(
        input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad
    )
    elapsed = time.perf_counter() - started
    new = out[0, input_ids.shape[-1]:]
    return {
        "text": tokenizer.decode(new, skip_special_tokens=True).strip(),
        "tokens": int(new.numel()),
        "tokens_per_s": round(new.numel() / elapsed, 2) if elapsed > 0 else None,
    }


def run(args: argparse.Namespace, log=print) -> List[Dict[str, Any]]:
    alphas = parse_alphas(args.alphas)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = sorted(set(modes) - set(MODES))
    if unknown or not modes:
        raise CalibrationDataError(f"--modes: {unknown or 'vacío'}; usa additive, skop o ambos")
    requested = parse_layers(args.layers)
    if not args.calibration.is_file():
        raise CalibrationDataError(f"{args.calibration}: no existe")

    log(f"Cargando modelo {'tiny (--dev)' if args.dev else args.model} [{'dev' if args.dev else args.quant}]...")
    model, tokenizer, _ = _load_model(args)
    engine = SKOPEngine(model, SKOPConfig(), tokenizer=tokenizer).load(str(args.calibration))
    layers = engine.layer_ids if requested is None else requested
    missing = sorted(set(layers) - set(engine.layer_ids))
    if missing:
        raise CalibrationDataError(f"--layers: {missing} no están en la calibración (capas {engine.layer_ids})")
    span = f"{layers[0]}–{layers[-1]}" if layers == list(range(layers[0], layers[-1] + 1)) else ",".join(map(str, layers))
    chat = not args.no_chat

    records: List[Dict[str, Any]] = []

    def emit(record: Dict[str, Any], header: str) -> None:
        records.append(record)
        log(f"\n== {header} · {record['tokens']} tokens · {record['tokens_per_s']} tok/s\n{record['text']}")

    engine.detach()
    emit({"mode": "base", "alpha": 0.0, "layers": [], **_generate(engine, tokenizer, args.prompt, chat, args.max_new_tokens)},
         "base (sin steering)")
    for mode in modes:
        projection = mode == "skop"
        for alpha in alphas:
            if alpha == 0:
                continue  # idéntico a la base
            engine.set_layer_plan({l: (alpha, projection) for l in layers})
            projected = sum(int(engine.risk_mask[l].sum()) for l in layers) if projection else 0
            engine.attach()
            try:
                result = _generate(engine, tokenizer, args.prompt, chat, args.max_new_tokens)
            finally:
                engine.detach()
            emit({"mode": mode, "alpha": alpha, "layers": layers, "projected_heads": projected, **result},
                 f"{mode} α={alpha:+g} · capas {span}" + (f" · {projected} cabezas proyectadas" if projection else ""))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps({"prompt": args.prompt, **record}, ensure_ascii=False) + "\n")
        log(f"\nGuardado en {args.output}")
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (CalibrationDataError, ValueError, RuntimeError, OSError) as exc:
        print(f"latentflow-probe: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
