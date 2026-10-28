"""
latentflow.tools.evaluate
=========================

CLI ``latentflow-eval``: evaluación offline del steering SKOP frente al steering
vanilla, en el espíritu de los experimentos del paper (arXiv:2605.06342) y a escala
de una GPU de 8 GB. Para cada comportamiento y cada (modo, α) mide tres cosas:

1. **Steering:** preguntas A/B de las Model-Written Evaluations de Anthropic
   (Perez et al., 2022; https://github.com/anthropics/evals). La dirección ``r_q``
   del comportamiento se estima con preguntas de entrenamiento (respuesta que
   muestra el comportamiento menos la que no) y se puntúa en otras preguntas con la
   probabilidad que el modelo da a la opción que muestra el comportamiento (azar:
   0.5; las preguntas con más de dos opciones se descartan).
2. **Utilidad:** acierto en preguntas de opción múltiple (ARC-Easy), puntuando cada
   opción por su log-verosimilitud.
3. **Mecanismo:** cambio de la masa de atención sobre el Focus Set del modelo base
   (Ec. 8 del paper), ``ΔM = M_steered(F) − M_base(F)``, por (cabeza, paso) en
   textos de utilidad con teacher forcing. Se informa la media y la fracción de
   pares que pierden al menos 10, 15 y 25 puntos de masa.

Todo es local: los datos se pasan como ficheros y no se descarga nada. Diferencias
con el paper: puntuación por probabilidad de la opción en vez de juez LLM, ARC-Easy
como utilidad y teacher forcing en vez de pasos de decodificación para ΔM.

Uso::

    python -m latentflow.tools.evaluate --calibration calib/gemma2_skop.pt \\
        --behavior data/anthropic-evals/advanced-ai-risk/human_generated_evals/corrigible-less-HHH.jsonl \\
        --utility data/arc_easy_test.jsonl --alphas=-3,-2,-1,1,2,3 -o results/eval.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from latentflow._console import ensure_utf8_stdio
from latentflow.core.steering import DEFAULT_MODEL_ID, SKOPConfig, SKOPEngine, focus_set_mask
from latentflow.tools.initializer import CalibrationDataError, _load_model, parse_layers
from latentflow.tools.probe import MODES, parse_alphas

__all__ = [
    "ChoiceQuestion",
    "MassProbe",
    "build_parser",
    "load_arc",
    "load_mwe",
    "main",
    "option_logprobs",
    "run",
]

_LETTER = re.compile(r"\(([A-Z])\)")
MASS_DROPS = (0.10, 0.15, 0.25)
CSV_FIELDS = [
    "behavior", "mode", "alpha", "layers", "projected_heads",
    "steer_p_match", "steer_match_rate", "util_acc", "util_p_correct",
    "mass_mean", "mass_drop_10", "mass_drop_15", "mass_drop_25", "mass_steps",
    "n_train", "n_test", "n_utility", "seconds",
]  # fmt: skip


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChoiceQuestion:
    """Pregunta de opción múltiple ya formateada (``Choices: (A) … (B) …``)."""

    text: str
    letters: Tuple[str, ...]
    answer: int  # índice de la opción correcta (ARC) o que muestra el comportamiento (MWE)


def _records(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    if not path.is_file():
        raise CalibrationDataError(f"{path}: no existe o no es un fichero")
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationDataError(f"{path}:{lineno}: JSON no válido ({exc.msg})") from None


def load_mwe(path: Path) -> List[ChoiceQuestion]:
    """Model-Written Evaluations: ``{"question", "answer_matching_behavior", ...}``.

    La pregunta ya trae las opciones; ``answer`` es la que muestra el comportamiento.
    """
    out: List[ChoiceQuestion] = []
    for lineno, rec in _records(path):
        try:
            question = rec["question"].strip()
            match = _LETTER.search(rec["answer_matching_behavior"]).group(1)
        except (KeyError, AttributeError, TypeError):
            raise CalibrationDataError(f"{path}:{lineno}: se esperaba question y answer_matching_behavior '(X)'") from None
        letters = tuple(dict.fromkeys(_LETTER.findall(question)))
        if match not in letters or len(letters) < 2:
            raise CalibrationDataError(f"{path}:{lineno}: la pregunta no lista las opciones (A), (B)...")
        out.append(ChoiceQuestion(question, letters, letters.index(match)))
    if not out:
        raise CalibrationDataError(f"{path}: no contiene preguntas")
    return out


def load_arc(path: Path) -> List[ChoiceQuestion]:
    """ARC (``allenai/ai2_arc`` exportado a JSONL): ``{"question", "choices": {"text", "label"}, "answerKey"}``.

    Las opciones se renombran A, B, C… por posición (algunas preguntas usan 1-4).
    """
    out: List[ChoiceQuestion] = []
    for lineno, rec in _records(path):
        try:
            texts, labels = list(rec["choices"]["text"]), [str(x) for x in rec["choices"]["label"]]
            answer = labels.index(str(rec["answerKey"]))
            question = rec["question"].strip()
        except (KeyError, ValueError, TypeError):
            raise CalibrationDataError(f"{path}:{lineno}: se esperaba question, choices {{text, label}} y answerKey") from None
        letters = tuple(chr(ord("A") + i) for i in range(len(texts)))
        body = "\n".join(f" ({letter}) {text}" for letter, text in zip(letters, texts))
        out.append(ChoiceQuestion(f"{question}\n\nChoices:\n{body}", letters, answer))
    if not out:
        raise CalibrationDataError(f"{path}: no contiene preguntas")
    return out


# --------------------------------------------------------------------------- #
# Tokenización y puntuación
# --------------------------------------------------------------------------- #
class _Prompter:
    """Plantilla de chat (si el tokenizer la tiene) y tokenización sin padding."""

    def __init__(self, tokenizer: Any, chat: bool = True) -> None:
        self.tokenizer = tokenizer
        self.chat = chat and bool(getattr(tokenizer, "chat_template", None))
        pad = getattr(tokenizer, "pad_token_id", None)
        self.pad_id = int(pad if pad is not None else tokenizer.eos_token_id)

    def prompt(self, user_text: str) -> str:
        if self.chat:
            messages = [{"role": "user", "content": user_text}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{user_text}\n\nAnswer: "

    def ids(self, text: str) -> List[int]:
        # La plantilla de chat ya incluye <bos>: no se añade otra vez.
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=not self.chat)
        return enc["input_ids"][0].tolist()

    def batch(self, sequences: Sequence[Sequence[int]]) -> Dict[str, Tensor]:
        """Padding a la derecha: las posiciones de cada secuencia empiezan en 0."""
        width = max(len(s) for s in sequences)
        input_ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros(len(sequences), width, dtype=torch.long)
        for row, seq in enumerate(sequences):
            input_ids[row, : len(seq)] = torch.tensor(list(seq), dtype=torch.long)
            mask[row, : len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


@torch.no_grad()
def option_logprobs(model: Any, prompter: _Prompter, prompt: str, options: Sequence[str], device: torch.device) -> List[float]:
    """Log-verosimilitud de cada continuación ``options[i]`` tras ``prompt`` (un forward por lote).

    Se puntúan los tokens de ``prompt + opción`` que difieren de los de ``prompt``,
    así que funciona aunque el tokenizer fusione el final del prompt con la opción.
    """
    base = prompter.ids(prompt)
    sequences, starts = [], []
    for option in options:
        full = prompter.ids(prompt + option)
        k = 0
        while k < min(len(base), len(full) - 1) and base[k] == full[k]:
            k += 1
        sequences.append(full)
        starts.append(max(k, 1))
    batch = prompter.batch(sequences)
    out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
    logits = out.logits if hasattr(out, "logits") else out
    scores = []
    for row, (seq, start) in enumerate(zip(sequences, starts)):
        target = torch.tensor(seq[start:], device=logits.device)
        logp = logits[row, start - 1 : len(seq) - 1].float().log_softmax(-1)
        scores.append(float(logp.gather(-1, target[:, None]).sum()))
    return scores


def _choice_scores(model: Any, prompter: _Prompter, questions: Sequence[ChoiceQuestion], device: torch.device) -> Tuple[float, float]:
    """Media de la probabilidad (normalizada entre opciones) de ``answer`` y tasa de acierto."""
    probs, hits = [], 0
    for q in questions:
        scores = torch.tensor(option_logprobs(model, prompter, prompter.prompt(q.text), [f"({l})" for l in q.letters], device))
        p = scores.softmax(-1)
        probs.append(float(p[q.answer]))
        hits += int(int(p.argmax()) == q.answer)
    return sum(probs) / len(probs), hits / len(questions)


# --------------------------------------------------------------------------- #
# Masa del Focus Set (Ec. 8)
# --------------------------------------------------------------------------- #
class MassProbe:
    """ΔM = M_steered(F) − M_base(F) por (cabeza, paso), con F el Focus Set del modelo base.

    ``prepare`` hace un forward por texto sin steering y guarda en CPU el Focus Set y
    su masa en cada capa; ``shift`` repite los forwards con el circuito que esté
    activo en el motor. Sólo cuentan los pasos con Focus y Tail no vacíos.
    """

    def __init__(self, engine: SKOPEngine, tau_high: float) -> None:
        self.engine = engine
        self.tau_high = tau_high
        self._inputs: List[Dict[str, Tensor]] = []
        self._base: List[Dict[int, Tuple[Tensor, Tensor, Tensor]]] = []  # capa -> (focus, masa, válido)

    def _attention(self, inputs: Dict[str, Tensor]) -> Dict[int, Tensor]:
        engine, captured = self.engine, {}

        def make_hook(layer: int):
            def hook(module: Any, args: Any, output: Any) -> None:
                weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if not isinstance(weights, Tensor):
                    raise RuntimeError("La atención no devuelve pesos: carga el modelo con attn_implementation='eager'")
                captured[layer] = weights[0].detach()  # (H, T, T)

            return hook

        handles = [engine.decoder_layers[l].self_attn.register_forward_hook(make_hook(l)) for l in engine.layer_ids]
        try:
            device = engine._input_device()
            with torch.no_grad():
                engine.model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
        finally:
            for h in handles:
                h.remove()
        return captured

    def prepare(self, inputs: Sequence[Dict[str, Tensor]]) -> "MassProbe":
        if self.engine.is_attached:
            raise RuntimeError("MassProbe.prepare necesita el motor sin steering (detach)")
        self._inputs, self._base = list(inputs), []
        for item in self._inputs:
            per_layer = {}
            for l, probs in self._attention(item).items():
                attendable = probs > 0
                focus = focus_set_mask(probs, attendable, self.tau_high)
                valid = focus.any(-1) & (attendable & ~focus).any(-1)
                mass = (probs.float() * focus).sum(-1)
                per_layer[l] = (focus.cpu(), mass.cpu(), valid.cpu())
            self._base.append(per_layer)
        return self

    def shift(self) -> Tensor:
        """ΔM de todos los (capa, cabeza, paso) válidos, en float64/CPU."""
        deltas = []
        for item, base in zip(self._inputs, self._base):
            for l, probs in self._attention(item).items():
                focus, mass, valid = base[l]
                steered = (probs.float().cpu() * focus).sum(-1)
                deltas.append((steered - mass)[valid].to(torch.float64))
        return torch.cat(deltas) if deltas else torch.zeros(0, dtype=torch.float64)


def mass_summary(delta: Tensor) -> Dict[str, float]:
    if delta.numel() == 0:
        return {"mass_mean": 0.0, **{f"mass_drop_{int(x * 100)}": 0.0 for x in MASS_DROPS}, "mass_steps": 0}
    return {
        "mass_mean": float(delta.mean()),
        **{f"mass_drop_{int(x * 100)}": float((delta <= -x).double().mean()) for x in MASS_DROPS},
        "mass_steps": int(delta.numel()),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latentflow-eval",
        description="Steering (MWE A/B), utilidad (ARC) y masa del Focus Set para un barrido de α, vanilla frente a SKOP.",
    )
    data = parser.add_argument_group("datos (locales, .jsonl)")
    data.add_argument("--calibration", type=Path, required=True, help="artefacto de latentflow-calibrate (Σ_ΔK)")
    data.add_argument("--behavior", type=Path, action="append", required=True, metavar="PATH",
                      help="fichero de Model-Written Evaluations; repetible")
    data.add_argument("--utility", type=Path, required=True, help="preguntas ARC en JSONL")
    data.add_argument("--n-train", type=int, default=100, help="preguntas para estimar r_q de cada comportamiento")
    data.add_argument("--n-test", type=int, default=100, help="preguntas para puntuar el steering")
    data.add_argument("--n-utility", type=int, default=200, help="preguntas de utilidad")
    data.add_argument("--n-mass", type=int, default=30, help="textos para la masa del Focus Set")
    data.add_argument("--seed", type=int, default=0)

    sweep = parser.add_argument_group("barrido")
    sweep.add_argument("--alphas", default="-3,-2,-1,1,2,3", help="intensidades (usa --alphas=-3,... si empieza por '-')")
    sweep.add_argument("--modes", default="additive,skop")
    sweep.add_argument("--layers", default=None, help="capas con steering, p. ej. '4-12' (por defecto todas)")
    sweep.add_argument("--no-chat", action="store_true", help="no aplicar la plantilla de chat")
    sweep.add_argument("-o", "--output", type=Path, default=Path("results/eval.csv"))

    model = parser.add_argument_group("modelo")
    model.add_argument("--model", default=DEFAULT_MODEL_ID)
    model.add_argument("--quant", choices=("4bit", "8bit", "none"), default="4bit")
    model.add_argument("--dev", action="store_true", help="modelo tiny de pesos aleatorios (pruebas)")
    model.add_argument("--dev-layers", type=int, default=26)
    return parser


def _split(questions: List[ChoiceQuestion], n_train: int, n_test: int, seed: int) -> Tuple[List[ChoiceQuestion], List[ChoiceQuestion]]:
    shuffled = questions[:]
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < n_train + n_test:
        raise CalibrationDataError(f"hay {len(shuffled)} preguntas y se piden {n_train} + {n_test}")
    return shuffled[:n_train], shuffled[n_train : n_train + n_test]


def _direction_batches(prompter: _Prompter, questions: Sequence[ChoiceQuestion], batch_size: int):
    """Pares (respuesta que muestra el comportamiento, la que no), cortados en la letra."""
    for start in range(0, len(questions), batch_size):
        chunk = questions[start : start + batch_size]
        pos, neg = [], []
        for q in chunk:
            other = next(i for i in range(len(q.letters)) if i != q.answer)
            prompt = prompter.prompt(q.text)
            pos.append(prompter.ids(prompt + f"({q.letters[q.answer]}"))
            neg.append(prompter.ids(prompt + f"({q.letters[other]}"))
        yield prompter.batch(pos), prompter.batch(neg)


def run(args: argparse.Namespace, log=print) -> List[Dict[str, Any]]:
    alphas = [a for a in parse_alphas(args.alphas) if a != 0]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes or set(modes) - set(MODES):
        raise CalibrationDataError(f"--modes: usa additive, skop o ambos (recibido {args.modes!r})")
    requested = parse_layers(args.layers)
    behaviors = {}
    for path in args.behavior:
        questions = load_mwe(path)
        # Sólo preguntas A/B: el azar es 0.5 en todos los comportamientos y el par
        # (muestra el comportamiento, no lo muestra) para r_q no es ambiguo.
        behaviors[path.stem] = [q for q in questions if len(q.letters) == 2]
        if len(behaviors[path.stem]) < len(questions):
            log(f"{path.stem}: {len(behaviors[path.stem])} de {len(questions)} preguntas son A/B; se usan sólo esas")
    utility = load_arc(args.utility)
    random.Random(args.seed).shuffle(utility)
    utility = utility[: args.n_utility]
    splits = {name: _split(qs, args.n_train, args.n_test, args.seed) for name, qs in behaviors.items()}
    if not args.calibration.is_file():
        raise CalibrationDataError(f"{args.calibration}: no existe")

    log(f"Cargando modelo {'tiny (--dev)' if args.dev else args.model} [{'dev' if args.dev else args.quant}]...")
    model, tokenizer, _ = _load_model(args)
    engine = SKOPEngine(model, SKOPConfig(), tokenizer=tokenizer).load(str(args.calibration))
    layers = engine.layer_ids if requested is None else requested
    if set(layers) - set(engine.layer_ids):
        raise CalibrationDataError(f"--layers: {sorted(set(layers) - set(engine.layer_ids))} no están en la calibración")
    prompter = _Prompter(tokenizer, chat=not args.no_chat)
    device = engine._input_device()
    span = f"{layers[0]}–{layers[-1]}" if layers == list(range(layers[0], layers[-1] + 1)) else ",".join(map(str, layers))

    engine.detach()
    mass_inputs = [prompter.batch([prompter.ids(prompter.prompt(q.text))]) for q in utility[: args.n_mass]]
    probe = MassProbe(engine, engine.config.tau_high).prepare(mass_inputs)
    started = time.perf_counter()
    util_p, util_acc = _choice_scores(model, prompter, utility, device)
    log(f"Base: utilidad ARC {util_acc:.3f} (p {util_p:.3f}) en {len(utility)} preguntas · {time.perf_counter() - started:.0f} s")

    rows: List[Dict[str, Any]] = []
    common = {"layers": span, "n_train": args.n_train, "n_test": args.n_test, "n_utility": len(utility)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, (train, test) in splits.items():
        engine.set_layer_plan(None)
        engine.fit(steering_vectors=engine.concept_direction(_direction_batches(prompter, train, engine.config.batch_size)))
        started = time.perf_counter()
        p_match, rate = _choice_scores(model, prompter, test, device)
        rows.append({"behavior": name, "mode": "base", "alpha": 0.0, "projected_heads": 0, "steer_p_match": p_match,
                     "steer_match_rate": rate, "util_acc": util_acc, "util_p_correct": util_p, **mass_summary(torch.zeros(0)),
                     **common, "seconds": round(time.perf_counter() - started, 1)})
        log(f"\n[{name}] base: p(comportamiento) {p_match:.3f} · tasa {rate:.3f}")
        for mode in modes:
            for alpha in alphas:
                started = time.perf_counter()
                engine.set_layer_plan({l: (alpha, mode == "skop") for l in layers})
                projected = sum(int(engine.risk_mask[l].sum()) for l in layers) if mode == "skop" else 0
                engine.attach()
                try:
                    p_match, rate = _choice_scores(model, prompter, test, device)
                    u_p, u_acc = _choice_scores(model, prompter, utility, device)
                    mass = mass_summary(probe.shift())
                finally:
                    engine.detach()
                row = {"behavior": name, "mode": mode, "alpha": alpha, "projected_heads": projected,
                       "steer_p_match": p_match, "steer_match_rate": rate, "util_acc": u_acc, "util_p_correct": u_p,
                       **mass, **common, "seconds": round(time.perf_counter() - started, 1)}
                rows.append(row)
                log(f"[{name}] {mode:8s} α={alpha:+g} · capas {span}: p(comp.) {p_match:.3f} · utilidad {u_acc:.3f} · "
                    f"ΔM medio {mass['mass_mean']:+.3f} · pierden ≥10 pts {mass['mass_drop_10']:.1%} · {row['seconds']} s")
                _write_csv(args.output, rows)  # resultados parciales: un corte no pierde lo hecho
    _write_csv(args.output, rows)
    log(f"\nResultados: {args.output} ({len(rows)} filas)")
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (CalibrationDataError, ValueError, RuntimeError, OSError) as exc:
        print(f"latentflow-eval: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
