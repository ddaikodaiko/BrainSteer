"""``latentflow-probe`` sobre el modelo dev: base, barrido de α y modos, JSONL."""

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from latentflow.tools.initializer import main as calibrate  # noqa: E402
from latentflow.tools.probe import main as probe  # noqa: E402
from latentflow.tools.probe import parse_alphas  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "calibration"
DEV = ["--dev", "--dev-layers", "3"]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    out = tmp_path_factory.mktemp("probe") / "skop.pt"
    args = ["--utility", str(EXAMPLES / "utility.example.jsonl"), "--concept", str(EXAMPLES / "formal_informal.jsonl"),
            *DEV, "--batch-size", "4", "--max-length", "48", "-o", str(out)]
    assert calibrate(args) == 0
    return out


def test_probe_sweeps_alphas_and_modes(artifact, tmp_path, capsys):
    out = tmp_path / "probe.jsonl"
    args = ["--calibration", str(artifact), "--prompt", "hola mundo", "--alphas", "0,3,-3",
            "--modes", "additive,skop", "--max-new-tokens", "4", "-o", str(out), *DEV]
    assert probe(args) == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["mode"], r["alpha"]) for r in records] == [
        ("base", 0.0), ("additive", 3.0), ("additive", -3.0), ("skop", 3.0), ("skop", -3.0)]
    assert all(r["prompt"] == "hola mundo" and r["tokens"] >= 1 for r in records)
    assert all(r["layers"] == [0, 1, 2] for r in records[1:])
    assert all(r["projected_heads"] > 0 for r in records if r["mode"] == "skop")
    log = capsys.readouterr().out
    assert "== base (sin steering)" in log and "== skop α=+3 · capas 0–2" in log


def test_probe_layer_subset_and_errors(artifact, tmp_path, capsys):
    out = tmp_path / "subset.jsonl"
    base = ["--calibration", str(artifact), "--prompt", "hola", "--max-new-tokens", "2", *DEV]
    assert probe([*base, "--alphas", "5", "--modes", "additive", "--layers", "1", "-o", str(out)]) == 0
    assert [json.loads(line)["layers"] for line in out.read_text(encoding="utf-8").splitlines()] == [[], [1]]
    assert probe([*base, "--layers", "7"]) == 2
    assert "no están en la calibración" in capsys.readouterr().err
    assert probe([*base, "--modes", "lateral"]) == 2
    assert "--modes" in capsys.readouterr().err


def test_parse_alphas():
    assert parse_alphas("0, 4,-4.5") == [0.0, 4.0, -4.5]
    with pytest.raises(ValueError):
        parse_alphas("cuatro")
