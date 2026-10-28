"""Tests de integración del ciclo de vida de ``/ws/stream``.

Usan el backend de desarrollo (``latentflow.tools.dev_backend``): un
``LlamaForCausalLM`` real de Transformers con pesos aleatorios, sin descargas.
``generate()``, el streamer, los hooks y el worker son los de producción; sólo
cambian el tamaño del modelo y el tokenizer.
"""

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

fastapi = pytest.importorskip("fastapi")
transformers = pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from latentflow.api.server import DEFAULT_ALLOWED_ORIGINS, OriginPolicy, ServerSettings, create_app  # noqa: E402
from latentflow.api.server import main as server_main  # noqa: E402
from latentflow.core.sae import GemmaScopeResolver, HashFeatureResolver  # noqa: E402
from latentflow.tools.dev_backend import (  # noqa: E402
    CharTokenizer,
    DevModelSpec,
    _parse_args,
    build_dev_runtime,
    create_dev_app,
    dev_runtime_factory,
)
from latentflow.tools.dev_backend import main as dev_main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VOCAB, N_LAYERS, N_HEADS = 100, 4, 8
SETTINGS = ServerSettings(model_id="tiny-llama-test", load_in_4bit=False, alpha=2.0)


def make_runtime_factory(delay: float = 0.0, calibrated: bool = True):
    """Mismo backend que `latentflow-dev`, con una geometría más pequeña."""
    return dev_runtime_factory(
        DevModelSpec(
            num_layers=N_LAYERS, num_heads=N_HEADS, num_kv_heads=2, head_dim=16, hidden_size=64,
            intermediate_size=128, vocab_size=VOCAB, delay=delay, calibrated=calibrated,
            calibration_batches=4, calibration_seq_len=6,
        )
    )


@contextmanager
def serve(delay: float = 0.0, calibrated: bool = True, settings: ServerSettings = SETTINGS):
    with TestClient(create_app(settings, make_runtime_factory(delay, calibrated))) as client:
        yield client


@pytest.fixture
def client():
    with serve() as c:
        yield c


def recv_until(ws, predicate, limit=5000):
    messages = []
    for _ in range(limit):
        msg = ws.receive_json()
        messages.append(msg)
        if predicate(msg):
            return messages
    raise AssertionError(f"No llegó el mensaje esperado; recibidos: {[m['type'] for m in messages]}")


def of_type(messages, kind):
    return [m for m in messages if m["type"] == kind]


def generate(ws, **kwargs):
    payload = {"action": "generate", "prompt": "hola mundo", "chat": False, **kwargs}
    ws.send_json(payload)
    return recv_until(ws, lambda m: m["type"] == "end")


# --------------------------------------------------------------------------- #
# Conexión y estado
# --------------------------------------------------------------------------- #
def test_health_reports_calibrated_circuit(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["circuit"]["calibrated"] is True
    assert body["circuit"]["steering"] is True  # steering_on_start
    assert body["circuit"]["num_risky_heads"] > 0


def test_ready_message_and_status(client):
    with client.websocket_connect("/ws/stream") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["geometry"] == {"num_layers": N_LAYERS, "num_heads": N_HEADS, "num_kv_heads": 2, "head_dim": 16}
        assert ready["circuit"]["skop_filter"] is True
        ws.send_json({"action": "status"})
        status = ws.receive_json()
        assert status["type"] == "status" and status["sessions"] == 1 and status["active_request_id"] is None


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
def test_generate_streams_tokens_with_risk_telemetry(client):
    runtime = client.app.state.runtime
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        msgs = generate(ws, max_new_tokens=6, request_id="req-1")

    assert [m["type"] for m in msgs] == ["accepted", "start"] + ["token"] * 6 + ["end"]
    assert all(m["request_id"] == "req-1" for m in msgs)
    tokens = of_type(msgs, "token")
    assert [t["index"] for t in tokens] == list(range(6))

    risky = {(l, h) for l in runtime.engine.layer_ids for h in range(N_HEADS) if runtime.engine.risk_mask[l][h]}
    for t in tokens:
        tel = t["telemetry"]
        assert tel["available"] is True
        # Energía absoluta de Rayleigh (SKOP): no negativa, sin cota superior fija.
        assert 0.0 <= tel["summary"]["mean"] <= tel["summary"]["max"]
        assert {(h["layer"], h["head"]) for h in tel["heads"]} == risky
        assert all(h["projected"] and h["risky"] and h["rayleigh"] >= 0.0 for h in tel["heads"])

    end = msgs[-1]
    assert end["reason"] == "length" and end["num_tokens"] == 6
    ids = [t["token_id"] for t in tokens]
    assert "".join(t["token"] for t in tokens) == end["text"] == runtime.tokenizer.decode(ids)


@pytest.mark.parametrize("mode,expect_heads", [("none", None), ("summary", 0), ("all", N_LAYERS * N_HEADS)])
def test_telemetry_modes(client, mode, expect_heads):
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        token = of_type(generate(ws, max_new_tokens=2, telemetry=mode), "token")[0]
    tel = token["telemetry"]
    if expect_heads is None:
        assert tel is None
    elif expect_heads == 0:
        assert "summary" in tel and "heads" not in tel
    else:
        assert len(tel["heads"]) == expect_heads


# --------------------------------------------------------------------------- #
# Hot-swapping del circuito
# --------------------------------------------------------------------------- #
def test_hot_swap_when_idle_mutates_engine_and_telemetry(client):
    engine = client.app.state.runtime.engine
    with client.websocket_connect("/ws/stream") as ws:
        v0 = ws.receive_json()["circuit"]["version"]
        ws.send_json({"action": "update", "alpha": 3.5, "skop_filter": False})
        accepted, circuit = ws.receive_json(), ws.receive_json()
        assert accepted == {"type": "update_accepted", "changes": {"alpha": 3.5, "skop_filter": False}, "deferred": False}
        assert circuit["type"] == "circuit" and circuit["applied_at_index"] is None
        assert circuit["circuit"]["version"] == v0 + 1
        assert engine.alpha == 3.5 and engine.use_projection is False
        for l, v in engine.effective_vectors().items():
            torch.testing.assert_close(v, 3.5 * engine.steering_vectors[l])

        token = of_type(generate(ws, max_new_tokens=2), "token")[0]
        assert token["circuit_version"] == v0 + 1
        assert not any(h["projected"] for h in token["telemetry"]["heads"])

        ws.send_json({"action": "update", "steering": False})
        recv_until(ws, lambda m: m["type"] == "circuit")
        assert not engine.is_attached


def test_hot_swap_during_generation_applies_at_token_boundary():
    with serve(delay=0.02) as client:
        engine = client.app.state.runtime.engine
        with client.websocket_connect("/ws/stream") as ws:
            v0 = ws.receive_json()["circuit"]["version"]
            ws.send_json({"action": "generate", "prompt": "hola", "chat": False, "max_new_tokens": 30})
            head = recv_until(ws, lambda m: m["type"] == "token")
            ws.send_json({"action": "update", "alpha": 7.0})
            msgs = head + recv_until(ws, lambda m: m["type"] == "end")

    assert of_type(msgs, "update_accepted")[0]["deferred"] is True
    circuits = of_type(msgs, "circuit")
    assert len(circuits) == 1, "el trabajo de apply encolado no debe duplicar el cambio"
    k = circuits[0]["applied_at_index"]
    assert isinstance(k, int) and 0 <= k < 29
    tokens = of_type(msgs, "token")
    assert all(t["circuit_version"] == v0 for t in tokens if t["index"] <= k)
    assert all(t["circuit_version"] == v0 + 1 for t in tokens if t["index"] > k)
    # El evento de circuito llega justo después del token frontera.
    pos = msgs.index(circuits[0])
    assert msgs[pos - 1]["type"] == "token" and msgs[pos - 1]["index"] == k
    assert engine.alpha == 7.0


def test_circuit_updates_are_broadcast_to_every_session(client):
    with client.websocket_connect("/ws/stream") as a, client.websocket_connect("/ws/stream") as b:
        a.receive_json(), b.receive_json()
        a.send_json({"action": "update", "risk_percentile": 0.5})
        recv_until(a, lambda m: m["type"] == "circuit")
        event = b.receive_json()
        assert event["type"] == "circuit" and event["changes"] == {"risk_percentile": 0.5}
        n_heads = N_LAYERS * N_HEADS
        assert event["circuit"]["num_risky_heads"] == pytest.approx(n_heads * 0.5, abs=1)


# --------------------------------------------------------------------------- #
# Cancelación, concurrencia y errores
# --------------------------------------------------------------------------- #
def test_cancel_stops_generation_and_session_is_reusable():
    with serve(delay=0.02) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"action": "generate", "prompt": "hola", "chat": False, "max_new_tokens": 500})
            recv_until(ws, lambda m: m["type"] == "token" and m["index"] == 1)
            ws.send_json({"action": "cancel"})
            end = recv_until(ws, lambda m: m["type"] == "end")[-1]
            assert end["reason"] == "cancelled" and end["num_tokens"] < 500
            assert generate(ws, max_new_tokens=2)[-1]["reason"] == "length"


def test_second_generate_in_same_session_is_rejected_as_busy():
    with serve(delay=0.02) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"action": "generate", "prompt": "a", "chat": False, "max_new_tokens": 20, "request_id": "r1"})
            recv_until(ws, lambda m: m["type"] == "token")
            ws.send_json({"action": "generate", "prompt": "b", "chat": False})
            msgs = recv_until(ws, lambda m: m["type"] == "end")
            errors = of_type(msgs, "error")
            assert errors and errors[0]["code"] == "busy" and errors[0]["request_id"] == "r1"


@pytest.mark.parametrize(
    "raw,code",
    [
        ("esto no es json", "invalid_message"),
        (json.dumps({"action": "volar"}), "invalid_message"),
        (json.dumps({"action": "update", "alpha": "mucho"}), "invalid_message"),
        (json.dumps({"action": "update", "alfa": 1.0}), "invalid_message"),  # extra="forbid"
        (json.dumps({"action": "update", "gamma_energy": 1.5}), "invalid_message"),
        (json.dumps({"action": "generate", "prompt": ""}), "invalid_message"),
        (json.dumps({"action": "update"}), "empty_update"),
        (json.dumps({"action": "cancel"}), "idle"),
    ],
)
def test_invalid_messages_return_errors_without_closing(client, raw, code):
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_text(raw)
        err = ws.receive_json()
        assert err["type"] == "error" and err["code"] == code
        ws.send_json({"action": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_invalid_message_names_the_failed_action_and_request(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_json({"action": "generate", "prompt": "hola", "request_id": "r9", "max_new_tokens": 0})
        err = ws.receive_json()
        assert (err["code"], err["action"], err["request_id"]) == ("invalid_message", "generate", "r9")
        assert "max_new_tokens" in err["message"] and err["details"]
        ws.send_text("[1, 2]")
        assert "action" not in ws.receive_json()


def test_disconnect_mid_generation_cancels_worker_job():
    with serve(delay=0.02) as client:
        runtime = client.app.state.runtime
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"action": "generate", "prompt": "hola", "chat": False, "max_new_tokens": 1000})
            recv_until(ws, lambda m: m["type"] == "token")
        # 1000 tokens × 20 ms = 20 s si no se cancelara.
        deadline = time.monotonic() + 5.0
        while runtime.busy and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not runtime.busy
        assert not runtime.sessions


def test_uncalibrated_runtime_streams_without_telemetry_and_can_calibrate():
    with serve(calibrated=False) as client:
        engine = client.app.state.runtime.engine
        with client.websocket_connect("/ws/stream") as ws:
            assert ws.receive_json()["circuit"]["calibrated"] is False
            ws.send_json({"action": "update", "alpha": 2.0})
            assert ws.receive_json()["code"] == "not_calibrated"
            token = of_type(generate(ws, max_new_tokens=2), "token")[0]
            assert token["telemetry"] == {"available": False}

            ws.send_json({
                "action": "calibrate",
                "utility": ["Explica la gravedad a un niño de diez años.", "Calcula 17 por 23 paso a paso."],
                "positive": ["sé amable", "ayuda"],
                "negative": ["sé cruel", "ignora"],
            })
            msgs = recv_until(ws, lambda m: m["type"] in ("calibrated", "error"))
            assert msgs[-1]["type"] == "calibrated", msgs[-1]
            assert msgs[-1]["circuit"]["calibrated"] is True
            assert (engine.num_concept_pairs, engine.num_utility_samples) == (2, 2) and engine.num_delta_steps > 0
            ws.send_json({"action": "update", "steering": True})
            circuit = recv_until(ws, lambda m: m["type"] == "circuit" and "steering" in m["changes"])[-1]
            assert circuit["circuit"]["steering"] is True and engine.is_attached


# --------------------------------------------------------------------------- #
# asyncio + servidor real: el bucle de eventos no se bloquea
# --------------------------------------------------------------------------- #
def test_event_loop_stays_responsive_while_streaming():
    uvicorn = pytest.importorskip("uvicorn")
    websockets = pytest.importorskip("websockets")

    async def recv_json(ws, timeout=10.0):
        return json.loads(await asyncio.wait_for(ws.recv(), timeout))

    async def until(ws, predicate):
        msgs = []
        while True:
            msgs.append(await recv_json(ws))
            if predicate(msgs[-1]):
                return msgs

    async def scenario():
        app = create_app(SETTINGS, make_runtime_factory(delay=0.03))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        serve_task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}/ws/stream"
        try:
            async with websockets.connect(url) as a, websockets.connect(url) as b:
                await recv_json(a), await recv_json(b)
                await a.send(json.dumps({"action": "generate", "prompt": "x", "chat": False, "max_new_tokens": 40}))
                await until(a, lambda m: m["type"] == "token")

                # Mismo socket: el ping se atiende a mitad del stream.
                await a.send(json.dumps({"action": "ping"}))
                t0 = time.perf_counter()
                mid = await until(a, lambda m: m["type"] in ("pong", "end"))
                assert mid[-1]["type"] == "pong", "el pong debe llegar antes de que acabe la generación"
                assert time.perf_counter() - t0 < 0.5

                # Otra sesión: su generación queda en cola detrás de la de `a`.
                await b.send(json.dumps({"action": "generate", "prompt": "y", "chat": False, "max_new_tokens": 5}))
                accepted = await until(b, lambda m: m["type"] == "accepted")
                assert accepted[-1]["queued"] is True

                end_a = (await until(a, lambda m: m["type"] == "end"))[-1]
                end_b = (await until(b, lambda m: m["type"] == "end"))[-1]
                assert (end_a["reason"], end_a["num_tokens"]) == ("length", 40)
                assert (end_b["reason"], end_b["num_tokens"]) == ("length", 5)
        finally:
            server.should_exit = True
            await serve_task

    asyncio.run(asyncio.wait_for(scenario(), timeout=90))


# --------------------------------------------------------------------------- #
# Plan por capa (DAG serializado por el frontend)
# --------------------------------------------------------------------------- #
def test_layer_plan_update_applies_per_layer_alpha_and_mode(client):
    engine = client.app.state.runtime.engine
    plan = [{"layer": 1, "alpha": 2.5, "mode": "orthogonal"}, {"layer": 3, "alpha": -1.0, "mode": "additive"}]
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_json({"action": "update", "steering": True, "layers": plan})
        circuit = recv_until(ws, lambda m: m["type"] == "circuit")[-1]["circuit"]
        assert circuit["layer_plan"] == plan and circuit["active_layers"] == [1, 3]
        assert engine.active_layers == [1, 3] and len(engine._handles) == 2

        heads = of_type(generate(ws, max_new_tokens=1, telemetry="all"), "token")[0]["telemetry"]["heads"]
        by_layer = {}
        for h in heads:
            by_layer.setdefault(h["layer"], []).append(h)
        assert all(h["steered"] for h in by_layer[1] + by_layer[3])
        assert not any(h["steered"] or h["projected"] for h in by_layer[0] + by_layer[2])
        assert all(h["projected"] == h["risky"] for h in by_layer[1])  # orthogonal
        assert not any(h["projected"] for h in by_layer[3])             # additive

        ws.send_json({"action": "update", "reset_layers": True})
        circuit = recv_until(ws, lambda m: m["type"] == "circuit")[-1]["circuit"]
        assert circuit["layer_plan"] is None and engine.active_layers == list(range(N_LAYERS))


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"layers": [{"layer": 99, "alpha": 1.0, "mode": "additive"}]}, "invalid_layer"),
        ({"layers": [{"layer": 1, "alpha": 1.0, "mode": "additive"}] * 2}, "invalid_message"),
        ({"layers": [{"layer": 1, "alpha": 1.0, "mode": "sideways"}]}, "invalid_message"),
        ({"layers": [], "reset_layers": True}, "invalid_message"),
    ],
)
def test_layer_plan_validation(client, payload, code):
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_json({"action": "update", **payload})
        assert ws.receive_json()["code"] == code


# --------------------------------------------------------------------------- #
# Seguridad: protección cross-site del WebSocket (Origin)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
        "tauri://localhost",
        "http://tauri.localhost",
        "HTTP://LocalHost:5173/",  # esquema/host sin distinguir mayúsculas, barra final
    ],
)
def test_frontend_origins_can_connect(client, origin):
    with client.websocket_connect("/ws/stream", headers={"origin": origin}) as ws:
        assert ws.receive_json()["type"] == "ready"


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "null",                                  # iframe sandbox / file://
        "http://localhost:5174",                 # otro puerto local
        "http://localhost",                      # sin puerto
        "https://localhost:5173",                # otro esquema
        "http://localhost.evil.com:5173",        # sufijo engañoso
        "http://127.0.0.1:5173.evil.com",
        "http://evil.example/http://localhost:5173",
        "",
    ],
)
def test_foreign_origins_are_rejected_before_any_session(client, origin):
    runtime = client.app.state.runtime
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream", headers={"origin": origin}):
            pass
    assert exc.value.code == 1008
    assert not runtime.sessions and not runtime.busy


def test_missing_origin_policy_is_configurable():
    strict = dataclasses.replace(SETTINGS, allow_missing_origin=False)
    with serve(settings=strict) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/stream"):
                pass
        with client.websocket_connect("/ws/stream", headers={"origin": "http://localhost:5173"}) as ws:
            assert ws.receive_json()["type"] == "ready"


def test_origin_policy_unit():
    policy = OriginPolicy(["http://localhost:5173"], allow_missing=False)
    assert policy.is_allowed(["http://localhost:5173"])
    assert not policy.is_allowed([])
    assert not policy.is_allowed(["http://localhost:5173", "http://localhost:5173"])
    wildcard = OriginPolicy(["*"])
    assert wildcard.is_allowed(["https://anything.example"])
    assert not wildcard.is_allowed(["null"])


def test_importing_the_server_with_a_bad_env_is_safe():
    """#14: importar el módulo creaba la app y leía el entorno."""
    env = {**os.environ, "LATENTFLOW_ALPHA": "abc"}
    code = "import latentflow.api.server as s; assert 'app' not in vars(s); print('ok')"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr


@pytest.mark.parametrize("name,value,expected", [
    ("LATENTFLOW_ALPHA", "abc", "LATENTFLOW_ALPHA='abc' no es un número real"),
    ("LATENTFLOW_ALPHA", "nan", "fuera de rango"),
    ("LATENTFLOW_MAX_PROMPT_TOKENS", "1.5", "no es un número entero"),
    ("LATENTFLOW_MAX_PROMPT_TOKENS", "0", "mínimo 1"),
    ("LATENTFLOW_LOAD_IN_4BIT", "quizás", "no es un booleano"),
    ("LATENTFLOW_PORT", "70000", "1–65535"),
])  # fmt: skip
def test_server_cli_rejects_bad_env_before_loading_anything(monkeypatch, capsys, name, value, expected):
    import uvicorn

    monkeypatch.setenv(name, value)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("no debe arrancar"))
    assert server_main() == 2
    assert expected in capsys.readouterr().err


def test_server_cli_and_lazy_app_use_the_validated_env(monkeypatch):
    import uvicorn
    from fastapi import FastAPI

    import latentflow.api.server as server_module

    calls = []
    monkeypatch.setenv("LATENTFLOW_ALPHA", " 3.5 ")
    monkeypatch.setenv("LATENTFLOW_PORT", "9010")
    monkeypatch.setenv("LATENTFLOW_LOAD_IN_4BIT", "off")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    assert server_main() == 0
    app, kw = calls[0]
    assert isinstance(app, FastAPI) and kw == {"host": "127.0.0.1", "port": 9010}
    settings = ServerSettings.from_env()
    assert (settings.alpha, settings.load_in_4bit) == (3.5, False)
    monkeypatch.delitem(vars(server_module), "app", raising=False)
    assert isinstance(server_module.app, FastAPI)  # `uvicorn latentflow.api.server:app`
    assert server_module.app is server_module.app
    monkeypatch.delitem(vars(server_module), "app")


def test_allowed_origins_from_env(monkeypatch):
    monkeypatch.setenv("LATENTFLOW_ALLOWED_ORIGINS", " http://localhost:3000 , https://app.local ,")
    monkeypatch.setenv("LATENTFLOW_ALLOW_MISSING_ORIGIN", "0")
    settings = ServerSettings.from_env()
    assert settings.allowed_origins == ("http://localhost:3000", "https://app.local")
    assert settings.allow_missing_origin is False
    monkeypatch.delenv("LATENTFLOW_ALLOWED_ORIGINS")
    assert ServerSettings.from_env().allowed_origins == DEFAULT_ALLOWED_ORIGINS


def test_real_server_answers_403_to_foreign_origin():
    uvicorn = pytest.importorskip("uvicorn")
    websockets = pytest.importorskip("websockets")
    from websockets.exceptions import InvalidStatus

    async def scenario():
        app = create_app(SETTINGS, make_runtime_factory())
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        url = f"ws://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}/ws/stream"
        try:
            with pytest.raises(InvalidStatus) as exc:
                async with websockets.connect(url, origin="https://evil.example"):
                    pass
            assert exc.value.response.status_code == 403
            # Dos headers Origin (uno legítimo): ambiguo -> rechazo. La capa `websockets`
            # de uvicorn ya responde 400; OriginPolicy (403) cubre servidores ASGI que
            # lo dejen pasar (ver test_origin_policy_unit).
            dup = [("Origin", "http://localhost:5173"), ("Origin", "https://evil.example")]
            with pytest.raises(InvalidStatus) as exc:
                async with websockets.connect(url, additional_headers=dup):
                    pass
            assert exc.value.response.status_code in (400, 403)
            async with websockets.connect(url, origin="http://localhost:5173") as ws:
                assert json.loads(await asyncio.wait_for(ws.recv(), 10))["type"] == "ready"
        finally:
            server.should_exit = True
            await task

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


# --------------------------------------------------------------------------- #
# Backend de desarrollo (latentflow.tools.dev_backend)
# --------------------------------------------------------------------------- #
def test_dev_backend_default_geometry_matches_gemma2_layers():
    runtime = build_dev_runtime(SETTINGS, DevModelSpec(delay=0))
    g = runtime.engine.geometry
    assert (g.num_layers, g.num_heads) == (26, 8)
    assert runtime.engine.is_fitted and runtime.engine.layer_ids == list(range(26))


def test_dev_app_serves_with_its_label_and_env_security(monkeypatch):
    monkeypatch.setenv("LATENTFLOW_ALLOWED_ORIGINS", "http://localhost:3000")
    spec = DevModelSpec(num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8, hidden_size=32, delay=0)
    with TestClient(create_dev_app(spec)) as client:
        assert client.get("/health").json()["model"] == "dev/tiny-llama-2L (pesos aleatorios)"
        with client.websocket_connect("/ws/stream", headers={"origin": "http://localhost:3000"}) as ws:
            assert ws.receive_json()["geometry"]["num_layers"] == 2
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/stream", headers={"origin": "http://localhost:5173"}):
                pass


def test_dev_cli_rejects_bad_env(monkeypatch, capsys):
    import uvicorn

    monkeypatch.setenv("LATENTFLOW_ALPHA", "abc")
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("no debe arrancar"))
    assert dev_main(["--delay", "0"]) == 2
    assert "latentflow-dev: error: LATENTFLOW_ALPHA='abc'" in capsys.readouterr().err


def test_dev_cli_args_and_tokenizer():
    args = _parse_args(["--layers", "4", "--delay", "0", "--port", "9001", "--uncalibrated"])
    assert (args.layers, args.delay, args.port, args.uncalibrated, args.host) == (4, 0.0, 9001, True, "127.0.0.1")
    enc = CharTokenizer(50)(["ab", "abcd"])
    assert enc["input_ids"].shape == (2, 4) and enc["attention_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert int(enc["input_ids"].max()) < 50


# --------------------------------------------------------------------------- #
# Features de SAE (Gemma Scope) en el circuito
# --------------------------------------------------------------------------- #
FID = "gemma-2-2b/0-gemmascope-res-16k/12082"
HIDDEN_TEST = 64  # hidden_size del runtime de los tests


def feature_layers(alpha=2.5, fid=FID):
    return [
        {"layer": 1, "alpha": 1.0, "mode": "orthogonal",
         "features": [{"feature_id": fid, "alpha": alpha}, {"feature_id": None, "alpha": -1.0}]},
        {"layer": 3, "alpha": 0.5, "mode": "additive"},
    ]


def test_sae_feature_update_injects_mapped_decoder_direction(client):
    engine = client.app.state.runtime.engine
    with client.websocket_connect("/ws/stream") as ws:
        ready = ws.receive_json()
        assert ready["sae"] == {"kind": "dev-hash"} and ready["hidden_size"] == HIDDEN_TEST
        ws.send_json({"action": "update", "steering": True, "layers": feature_layers()})
        msgs = recv_until(ws, lambda m: m["type"] == "circuit")
        assert of_type(msgs, "update_accepted")[0]["features"] == [{"feature_id": FID, "source": "dev-hash", "sae_layer": 0}]
        plan = msgs[-1]["circuit"]["layer_plan"]
        assert plan[0]["features"] == [
            {"feature_id": FID, "alpha": 2.5, "source": "dev-hash", "sae_layer": 0},
            {"feature_id": None, "alpha": -1.0, "source": "calibrated"},
        ]
        assert "features" not in plan[1]

        # Dirección = Σ α_f · r_q(f), con r_q(f) = q_proj(input_layernorm(W_dec[f])).
        decoder = HashFeatureResolver().resolve(FID, HIDDEN_TEST).vector
        expected = 2.5 * engine.residual_to_query(1, decoder) - 1.0 * engine.steering_vectors[1]
        torch.testing.assert_close(engine.direction_for(1), expected)
        assert set(engine.layer_directions) == {1}

        heads = of_type(generate(ws, max_new_tokens=2, telemetry="all"), "token")[0]["telemetry"]["heads"]
    layer1 = [h for h in heads if h["layer"] == 1]
    assert all(h["steered"] for h in layer1)
    assert [h["baseline"] for h in layer1] == pytest.approx(engine.rayleigh[1].tolist(), abs=1e-6)
    assert all(h["projected"] == h["risky"] for h in layer1)  # P_ΔK sólo en cabezas de riesgo


def test_feature_slider_hot_swap_during_stream():
    with serve(delay=0.02) as client:
        engine = client.app.state.runtime.engine
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"action": "generate", "prompt": "hola", "chat": False, "max_new_tokens": 30})
            head = recv_until(ws, lambda m: m["type"] == "token")
            for alpha in (1.0, -3.0):  # dos posiciones del slider
                ws.send_json({"action": "update", "steering": True, "layers": feature_layers(alpha)})
            msgs = head + recv_until(ws, lambda m: m["type"] == "end")
    assert not of_type(msgs, "error")
    assert [m["changes"]["layers"][0]["features"][0]["alpha"] for m in of_type(msgs, "update_accepted")] == [1.0, -3.0]
    assert of_type(msgs, "circuit") and msgs[-1]["reason"] == "length" and msgs[-1]["num_tokens"] == 30
    decoder = HashFeatureResolver().resolve(FID, HIDDEN_TEST).vector
    torch.testing.assert_close(engine.direction_for(1), -3.0 * engine.residual_to_query(1, decoder) - engine.steering_vectors[1])


class SlowResolver(HashFeatureResolver):
    def resolve(self, feature_id, d_model):
        time.sleep(0.3)
        return super().resolve(feature_id, d_model)


def test_updates_keep_their_order_while_a_feature_is_being_read(client):
    runtime = client.app.state.runtime
    runtime.feature_resolver = SlowResolver()
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_json({"action": "update", "steering": True, "layers": feature_layers()})
        ws.send_json({"action": "update", "reset_layers": True})  # llega antes de que acabe la lectura
        msgs = recv_until(ws, lambda m: m["type"] == "circuit" and "reset_layers" in m["changes"])
    accepted = [m["changes"] for m in of_type(msgs, "update_accepted")]
    assert "layers" in accepted[0] and accepted[1] == {"reset_layers": True}
    assert runtime.engine.layer_plan is None and runtime.engine.layer_directions == {}


@pytest.mark.parametrize("slow", [False, True], ids=["hash", "lectura-lenta"])
def test_generate_right_after_a_feature_update_prefills_with_the_new_circuit(slow):
    """#3: el prefill usaba el circuito anterior mientras se resolvía el SAE."""
    with serve(delay=0.03) as client:
        if slow:
            client.app.state.runtime.feature_resolver = SlowResolver()
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"action": "update", "steering": True, "layers": feature_layers()})
            ws.send_json({"action": "generate", "prompt": "hola mundo", "chat": False, "max_new_tokens": 6})
            msgs = recv_until(ws, lambda m: m["type"] == "end")
    kinds = [m["type"] for m in msgs]
    assert not of_type(msgs, "error")
    assert kinds.index("update_accepted") < kinds.index("start")
    plan = of_type(msgs, "start")[0]["circuit"]["layer_plan"]
    assert plan[0]["features"][0]["feature_id"] == FID
    assert [m["applied_at_index"] for m in of_type(msgs, "circuit")] == [None]  # antes del prefill
    assert len({t["circuit_version"] for t in of_type(msgs, "token")}) == 1


def test_cancel_while_waiting_for_a_feature_ends_without_forwards(client):
    client.app.state.runtime.feature_resolver = SlowResolver()
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_json({"action": "update", "steering": True, "layers": feature_layers()})
        ws.send_json({"action": "generate", "prompt": "hola", "chat": False, "max_new_tokens": 5})
        ws.send_json({"action": "generate", "prompt": "otra", "chat": False})
        ws.send_json({"action": "cancel"})
        msgs = recv_until(ws, lambda m: m["type"] == "end")
    assert of_type(msgs, "error")[0]["code"] == "busy"
    assert (msgs[-1]["reason"], msgs[-1]["num_tokens"]) == ("cancelled", 0)
    assert not of_type(msgs, "token")


def test_gemma_scope_weights_are_read_from_local_disk(client, tmp_path):
    import numpy as np

    runtime = client.app.state.runtime
    runtime.feature_resolver = GemmaScopeResolver(str(tmp_path), hf_cache_dir=str(tmp_path / "hf"))
    with client.websocket_connect("/ws/stream") as ws:
        version = ws.receive_json()["circuit"]["version"]
        ws.send_json({"action": "update", "steering": True, "layers": feature_layers()})
        err = ws.receive_json()
        assert err["code"] == "feature_not_found" and "hf download" in err["message"]
        ws.send_json({"action": "update", "layers": feature_layers(fid="gemma-2-2b/0-gemmascope-att-16k/1")})
        assert ws.receive_json()["code"] == "feature_invalid"
        assert runtime.circuit.version == version  # nada aplicado

        sae_dir = tmp_path / "gemma-scope-2b-pt-res" / "layer_0" / "width_16k" / "average_l0_105"  # canonical de la capa 0
        sae_dir.mkdir(parents=True)
        w_dec = np.random.default_rng(0).standard_normal((20000, HIDDEN_TEST)).astype(np.float32)
        np.savez(sae_dir / "params.npz", W_dec=w_dec)
        ws.send_json({"action": "update", "steering": True, "layers": feature_layers()})
        msgs = recv_until(ws, lambda m: m["type"] in ("circuit", "error"))
    assert msgs[-1]["type"] == "circuit", msgs[-1]
    assert msgs[-1]["circuit"]["layer_plan"][0]["features"][0]["source"].startswith("gemma-scope:")
    engine = runtime.engine
    expected = 2.5 * engine.residual_to_query(1, torch.from_numpy(w_dec[12082])) - engine.steering_vectors[1]
    torch.testing.assert_close(engine.direction_for(1), expected)


def test_feature_spec_validation(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        bad = [{"layer": 1, "alpha": 1.0, "mode": "additive", "features": [{"feature_id": "", "alpha": 1.0}]}]
        ws.send_json({"action": "update", "layers": bad})
        assert ws.receive_json()["code"] == "invalid_message"
        # El simulador del dev backend acepta cualquier ID no vacío (también no-Neuronpedia).
        ws.send_json({"action": "update", "layers": feature_layers(fid="mi-feature-libre")})
        accepted = recv_until(ws, lambda m: m["type"] in ("update_accepted", "error"))[-1]
        assert accepted["type"] == "update_accepted"
        assert accepted["features"] == [{"feature_id": "mi-feature-libre", "source": "dev-hash", "sae_layer": None}]


def test_merged_pending_updates_keep_the_latest_layer_intent(client):
    """Dos cambios que caen en la misma frontera de token se fusionan sin invertir su orden."""
    circuit = client.app.state.runtime.circuit
    circuit.request({"steering": True, "layers": [{"layer": 1, "alpha": 2.0, "mode": "additive"}]})
    circuit.request({"reset_layers": True})
    assert circuit._pending == {"steering": True, "reset_layers": True}
    circuit.request({"layers": [{"layer": 2, "alpha": 1.0, "mode": "orthogonal"}]})
    assert "reset_layers" not in circuit._pending and circuit._pending["layers"][0]["layer"] == 2
    circuit._pending.clear()
