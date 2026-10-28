"""Ciclo de vida de ``latentflow-calibrate`` sin red: datos locales -> CLI ->
artefacto -> carga en frío del servidor con ``LATENTFLOW_CALIBRATION``."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from latentflow.core.steering import ARTIFACT_FORMAT  # noqa: E402
from latentflow.tools.dev_backend import DevModelSpec, create_dev_app  # noqa: E402
from latentflow.tools.initializer import (  # noqa: E402
    CalibrationDataError,
    clean_text,
    load_concept_pairs,
    load_utility_texts,
    main,
    parse_layers,
    utility_text,
)

ROOT = Path(__file__).resolve().parents[1]
DEV_LAYERS = 3


def write_jsonl(path: Path, records) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def datasets(tmp_path):
    utility = write_jsonl(
        tmp_path / "utility.jsonl",
        [
            {"instruction": "Explain gravity to a child.", "input": "", "output": "Things fall because the Earth pulls them."},
            {"question": "What is 17 * 23?", "answer": "17 * 23 = 391\n#### 391"},
            {"text": "Describe the water cycle in three steps."},
            "Write a short poem about autumn leaves.",
            {"text": "Describe the water cycle in three steps."},  # duplicado
            {"text": "   "},                                        # vacío tras limpiar
            {"text": "List two prime numbers <pad><pad> greater than ten."},
        ],
    )
    gsm8k = write_jsonl(tmp_path / "gsm8k.jsonl", [{"question": "If a book costs 12 and I pay 20, what is my change?", "answer": "8"}])
    concept = write_jsonl(
        tmp_path / "concept.jsonl",
        [
            {"positive": "Dear colleague, please review the report.", "negative": "hey look at this thing"},
            {"positive": "Thank you very much for your assistance.", "negative": "thx dude"},
            {"positive": "I would appreciate further details.", "negative": "tell me more i guess"},
        ],
    )
    return {"utility": utility, "gsm8k": gsm8k, "concept": concept}


def cli_args(datasets, output: Path, *extra: str):
    return [
        "--utility", str(datasets["utility"]), "--utility", str(datasets["gsm8k"]),
        "--concept", str(datasets["concept"]),
        "--dev", "--dev-layers", str(DEV_LAYERS), "--batch-size", "2", "--max-length", "48",
        "-o", str(output), *extra,
    ]  # fmt: skip


# --------------------------------------------------------------------------- #
# Ciclo de vida completo
# --------------------------------------------------------------------------- #
def test_cli_generates_artifact_that_server_cold_loads(datasets, tmp_path, monkeypatch, capsys):
    output = tmp_path / "calib" / "nested" / "dev_skop.pt"
    assert main(cli_args(datasets, output)) == 0
    log = capsys.readouterr().out
    assert "6 textos de utilidad, 3 pares de concepto" in log  # 5 únicos válidos + 1 de gsm8k
    assert output.is_file() and not output.with_name(output.name + ".tmp").exists()

    state = torch.load(output, map_location="cpu", weights_only=True)
    assert state["format"] == ARTIFACT_FORMAT
    assert state["layer_ids"] == list(range(DEV_LAYERS))
    assert state["num_utility_samples"] == 6 and state["num_concept_pairs"] == 3
    assert state["num_delta_pairs"] > 0
    g = state["geometry"]
    for l in state["layer_ids"]:
        assert state["sigma_delta_k"][l].shape == (g["num_kv_heads"], g["head_dim"], g["head_dim"])
        assert state["sigma_delta_k"][l].dtype == torch.float64
        assert state["steering_vectors"][l].shape == (g["num_heads"], g["head_dim"])
    meta = state["meta"]
    assert meta["model_id"].startswith("dev/tiny-llama-3L") and meta["quantization"] == "dev"
    assert meta["config"]["tau_high"] == 0.1 and len(meta["datasets"]["utility"]) == 2
    assert len(meta["datasets"]["concept"]["sha256"]) == 64

    # Carga en frío: el servidor arranca sin calibrar nada y usa el artefacto.
    monkeypatch.setenv("LATENTFLOW_CALIBRATION", str(output))
    spec = DevModelSpec(num_layers=DEV_LAYERS, delay=0.0, calibrated=False)
    with TestClient(create_dev_app(spec)) as client:
        engine = client.app.state.runtime.engine
        assert client.get("/health").json()["circuit"]["calibrated"] is True
        assert engine.metadata["tool"] == "latentflow-calibrate"
        for l in state["layer_ids"]:
            torch.testing.assert_close(engine.sigma_delta_k[l], state["sigma_delta_k"][l])
            torch.testing.assert_close(engine.steering_vectors[l], state["steering_vectors"][l])
        with client.websocket_connect("/ws/stream") as ws:
            assert ws.receive_json()["circuit"]["steering"] is True
            ws.send_json({"action": "generate", "prompt": "hola", "chat": False, "max_new_tokens": 3})
            msgs = []
            while not msgs or msgs[-1]["type"] != "end":
                msgs.append(ws.receive_json())
        tokens = [m for m in msgs if m["type"] == "token"]
        assert len(tokens) == 3 and all(t["telemetry"]["available"] for t in tokens)


def test_cli_refuses_to_overwrite_without_force(datasets, tmp_path, capsys):
    output = tmp_path / "skop.pt"
    output.write_bytes(b"artefacto previo")
    assert main(cli_args(datasets, output)) == 2
    assert "--force" in capsys.readouterr().err
    assert output.read_bytes() == b"artefacto previo"
    assert main(cli_args(datasets, output, "--force", "--layers", "0,2")) == 0
    assert torch.load(output, weights_only=True)["layer_ids"] == [0, 2]


def test_cli_runs_as_module_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "latentflow.tools.initializer", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0 and "latentflow-calibrate" in result.stdout


# --------------------------------------------------------------------------- #
# Errores de datos: mensaje claro y código 2, sin traceback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda d: d["utility"].unlink(), "no existe"),
        (lambda d: d["utility"].write_text('{"text": "ok"}\n{roto\n'), "utility.jsonl:2"),
        (lambda d: d["concept"].write_text('{"positive": "solo uno"}\n'), "concept.jsonl:1"),
        (lambda d: d["utility"].write_text('{"title": "sin campos"}\n'), "faltan los campos"),
    ],
)
def test_cli_reports_data_errors(datasets, tmp_path, capsys, mutate, expected):
    mutate(datasets)
    assert main(cli_args(datasets, tmp_path / "x.pt")) == 2
    captured = capsys.readouterr()
    assert expected in captured.err and "Traceback" not in captured.err
    assert "Cargando modelo" not in captured.out  # falla antes de cargar el modelo
    assert not (tmp_path / "x.pt").exists()


# --------------------------------------------------------------------------- #
# Carga y limpieza de datos
# --------------------------------------------------------------------------- #
def test_utility_formats():
    assert utility_text("hola") == "hola"
    assert utility_text({"text": "t"}) == "t"
    assert utility_text({"instruction": "I", "input": "", "output": "O"}) == "I\n\nO"
    assert utility_text({"question": "Q", "answer": "A"}) == "Q\n\nA"
    with pytest.raises(TypeError):
        utility_text(42)


def test_cleaning_removes_special_tokens_and_control_chars():
    assert clean_text("a<pad>b\x00c  \t d", ["<pad>"]) == "a bc d"
    assert clean_text("línea 1\nlínea 2") == "línea 1\nlínea 2"


def test_loaders_dedupe_sample_and_accept_json_lists(tmp_path):
    path = tmp_path / "u.json"
    path.write_text(json.dumps([f"texto {i}" for i in range(10)] + ["texto 3"]), encoding="utf-8")
    texts = load_utility_texts([path])
    assert len(texts) == 10
    sampled = load_utility_texts([path], limit=4, seed=1)
    assert sampled == load_utility_texts([path], limit=4, seed=1) and len(sampled) == 4
    assert sampled == sorted(sampled, key=texts.index)  # conserva el orden original

    concept = write_jsonl(tmp_path / "c.jsonl", [{"positive": "p", "negative": "n"}])
    assert load_concept_pairs(concept) == (["p"], ["n"])
    with pytest.raises(CalibrationDataError):
        load_utility_texts([write_jsonl(tmp_path / "e.jsonl", [{"text": ""}])])


def test_parse_layers():
    assert parse_layers(None) is None and parse_layers("all") is None
    assert parse_layers("0-3,8, 10-11") == [0, 1, 2, 3, 8, 10, 11]
    for bad in ("3-1", "x", "1-"):
        with pytest.raises(CalibrationDataError):
            parse_layers(bad)


def test_example_datasets_are_valid():
    examples = ROOT / "examples" / "calibration"
    assert len(load_utility_texts([examples / "utility.example.jsonl"])) == 7
    positive, negative = load_concept_pairs(examples / "concept.example.jsonl")
    assert len(positive) == len(negative) == 4
