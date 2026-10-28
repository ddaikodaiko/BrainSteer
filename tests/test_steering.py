from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latentflow.core.steering import (
    ARTIFACT_FORMAT,
    SecondMomentAccumulator,
    SKOPConfig,
    SKOPEngine,
    _aggregate_tokens,
    compute_orthogonal_projector,
    focus_tail_second_moment,
    rayleigh_quotient,
    select_risky_heads,
    trace_normalize,
)

F64 = torch.float64


def random_psd(d: int, rank: int, batch: tuple = (), seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(*batch, d, rank, generator=g, dtype=F64)
    return a @ a.mT


# --------------------------------------------------------------------------- #
# Proyector ortogonal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d,rank", [(8, 3), (64, 64), (128, 16), (256, 40)])
@pytest.mark.parametrize("gamma", [0.5, 0.90, 0.99, 1.0])
def test_projector_is_idempotent_and_symmetric(d, rank, gamma):
    P = compute_orthogonal_projector(random_psd(d, rank, seed=d + rank), gamma).projector
    assert P.dtype == F64
    assert torch.equal(P, P.mT), "P debe ser exactamente simétrica"
    torch.testing.assert_close(P @ P, P, rtol=0, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_projector_from_low_precision_sigma_is_still_exact(dtype):
    sigma = random_psd(64, 10, seed=1).to(dtype)
    P = compute_orthogonal_projector(sigma, 0.90).projector
    assert P.dtype == F64
    assert torch.equal(P, P.mT)
    torch.testing.assert_close(P @ P, P, rtol=0, atol=1e-10)


def test_projector_batched_over_heads():
    sigma = random_psd(32, 6, batch=(4,), seed=2)
    res = compute_orthogonal_projector(sigma, 0.90)
    assert res.projector.shape == (4, 32, 32)
    assert res.rank.shape == (4,)
    for h in range(4):
        single = compute_orthogonal_projector(sigma[h], 0.90)
        assert single.rank.item() == res.rank[h].item()
        torch.testing.assert_close(single.projector, res.projector[h], rtol=0, atol=1e-10)
    torch.testing.assert_close(res.projector @ res.projector, res.projector, rtol=0, atol=1e-10)


@pytest.mark.parametrize("gamma,expected_k", [(0.5, 1), (0.8, 2), (0.85, 3), (0.90, 3), (0.95, 4), (1.0, 5)])
def test_energy_threshold_selects_minimal_rank(gamma, expected_k):
    # Energía acumulada: 0.5, 0.8, 0.9, 0.95, 1.0
    sigma = torch.diag(torch.tensor([3.0, 5.0, 0.5, 1.0, 0.5], dtype=F64))
    res = compute_orthogonal_projector(sigma, gamma)
    assert res.rank.item() == expected_k
    assert torch.trace(res.projector).item() == pytest.approx(5 - expected_k)


def test_projector_annihilates_retained_subspace_and_preserves_complement():
    sigma = random_psd(48, 8, seed=3)
    res = compute_orthogonal_projector(sigma, 0.90)
    k = res.rank.item()
    U = res.eigenvectors[:, :k]
    W = res.eigenvectors[:, k:]
    torch.testing.assert_close(res.projector @ U, torch.zeros_like(U), rtol=0, atol=1e-10)
    torch.testing.assert_close(res.projector @ W, W, rtol=0, atol=1e-10)
    assert torch.all(res.eigenvalues[:-1] >= res.eigenvalues[1:]), "autovalores en orden descendente"


def test_zero_sigma_gives_identity():
    res = compute_orthogonal_projector(torch.zeros(16, 16), 0.90)
    assert res.rank.item() == 0
    assert torch.equal(res.projector, torch.eye(16, dtype=F64))


def test_invalid_gamma_raises():
    with pytest.raises(ValueError):
        compute_orthogonal_projector(torch.eye(4), 0.0)
    with pytest.raises(ValueError):
        SKOPConfig(gamma_energy=1.5)


# --------------------------------------------------------------------------- #
# Rayleigh y selección de riesgo
# --------------------------------------------------------------------------- #
def test_rayleigh_quotient_matches_definition_and_bounds():
    sigma = random_psd(16, 16, batch=(3,), seed=4)
    r = torch.randn(3, 16, dtype=F64)
    R = rayleigh_quotient(sigma, r)
    for h in range(3):
        expected = (r[h] @ sigma[h] @ r[h]) / (r[h] @ r[h])
        assert R[h].item() == pytest.approx(expected.item(), rel=1e-12)
        evals = torch.linalg.eigvalsh(sigma[h])
        assert evals[0] - 1e-9 <= R[h] <= evals[-1] + 1e-9
    # Invariante a la escala de r.
    torch.testing.assert_close(rayleigh_quotient(sigma, 7.5 * r), R)


def test_select_risky_heads_takes_top_20_percent():
    scores = torch.arange(100, dtype=F64).reshape(10, 10)
    mask, threshold = select_risky_heads(scores, 0.80)
    assert mask.sum().item() == 20
    assert mask.flatten()[80:].all() and not mask.flatten()[:80].any()
    assert 79 < threshold < 80


# --------------------------------------------------------------------------- #
# Acumulación y agregación
# --------------------------------------------------------------------------- #
def test_second_moment_accumulator_matches_direct_computation():
    acc = SecondMomentAccumulator(num_heads=2, head_dim=5)
    chunks = [torch.randn(n, 2, 5, dtype=torch.float16) for n in (3, 7, 1)]
    for c in chunks:
        acc.update(c)
    sigma = acc.finalize()
    full = torch.cat(chunks).to(F64)
    expected = torch.einsum("nhd,nhe->hde", full, full) / full.shape[0]
    assert sigma.device.type == "cpu" and sigma.dtype == F64
    torch.testing.assert_close(sigma, expected)
    assert torch.equal(sigma, sigma.mT)


def test_last_token_aggregation_handles_left_and_right_padding():
    x = torch.randn(1, 3, 2 * 4)
    right = torch.cat([x, torch.zeros(1, 2, 8)], dim=1)
    left = torch.cat([torch.zeros(1, 2, 8), x], dim=1)
    ref = _aggregate_tokens(x, torch.ones(1, 3), 2, 4, "last")
    out_r = _aggregate_tokens(right, torch.tensor([[1, 1, 1, 0, 0]]), 2, 4, "last")
    out_l = _aggregate_tokens(left, torch.tensor([[0, 0, 1, 1, 1]]), 2, 4, "last")
    torch.testing.assert_close(out_r, ref)
    torch.testing.assert_close(out_l, ref)
    mean_r = _aggregate_tokens(right, torch.tensor([[1, 1, 1, 0, 0]]), 2, 4, "mean")
    torch.testing.assert_close(mean_r, x.reshape(1, 3, 2, 4).mean(1))


# --------------------------------------------------------------------------- #
# Motor end-to-end con un modelo tipo Llama/Gemma diminuto (GQA)
# --------------------------------------------------------------------------- #
HIDDEN, N_HEADS, N_KV, HEAD_DIM, N_LAYERS, VOCAB = 32, 4, 2, 8, 3, 50


# Multiplica los logits: atención picuda como la de un modelo entrenado, para que
# haya keys por encima y por debajo de tau_high (Focus y Tail no vacíos).
SHARPNESS = 4.0


class TinyAttention(nn.Module):
    """Atención causal con GQA y padding que, como en Transformers (eager),
    devuelve ``(salida, pesos)``. ``return_weights=False`` imita SDPA (pesos None)."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, N_KV * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, N_KV * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.return_weights = True

    def forward(self, x, attention_mask=None):
        b, t, _ = x.shape
        rep = N_HEADS // N_KV
        q = self.q_proj(x).view(b, t, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(b, t, N_KV, HEAD_DIM).transpose(1, 2).repeat_interleave(rep, dim=1)
        v = self.v_proj(x).view(b, t, N_KV, HEAD_DIM).transpose(1, 2).repeat_interleave(rep, dim=1)
        scores = SHARPNESS * (q @ k.mT) / HEAD_DIM**0.5
        allowed = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        if attention_mask is not None:
            allowed = allowed & attention_mask.bool()[:, None, None, :]
        weights = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(-1)
        out = self.o_proj((weights @ v).transpose(1, 2).reshape(b, t, -1))
        return out, (weights if self.return_weights else None)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()

    def forward(self, x, attention_mask=None):
        return x + self.self_attn(x, attention_mask)[0]


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=N_HEADS,
            num_key_value_heads=N_KV,
            head_dim=HEAD_DIM,
            hidden_size=HIDDEN,
            num_hidden_layers=N_LAYERS,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.model.layers = nn.ModuleList(TinyLayer() for _ in range(N_LAYERS))
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h, attention_mask)
        return self.lm_head(h)


class CharTokenizer:
    pad_token, eos_token = None, "<eos>"

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=64):
        ids = [[1 + (ord(c) % (VOCAB - 1)) for c in t][:max_length] for t in texts]
        width = max(len(i) for i in ids)
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.long)
        for row, seq in enumerate(ids):
            input_ids[row, : len(seq)] = torch.tensor(seq)
            mask[row, : len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


def random_batch(n=6, t=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(0, VOCAB, (n, t), generator=g),
        "attention_mask": torch.ones(n, t, dtype=torch.long),
    }


def calibrate_tiny(engine, utility_batches=3, concept_batches=4):
    """Σ_ΔK con un utility set genérico y r_q con pares de concepto (independientes)."""
    for i in range(utility_batches):
        engine.accumulate_utility_batch(random_batch(n=4, t=10, seed=100 + i))
    for i in range(concept_batches):
        engine.accumulate_concept_batch(random_batch(seed=2 * i), random_batch(seed=2 * i + 1))
    return engine.finalize_calibration()


class probe_q:
    """Captura la salida de ``q_proj`` de una capa *después* del steering
    (el hook de steering se registra con prepend=True y corre antes)."""

    def __init__(self, engine, layer):
        self.module = engine.decoder_layers[layer].self_attn.q_proj
        self.value = None

    def __enter__(self):
        self.handle = self.module.register_forward_hook(lambda m, a, out: setattr(self, "value", out.detach().clone()))
        return self

    def __exit__(self, *exc):
        self.handle.remove()


@pytest.fixture
def calibrated_engine():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    engine = SKOPEngine(model, SKOPConfig(alpha=2.0, gamma_energy=0.90, risk_percentile=0.80))
    return calibrate_tiny(engine).fit()


def test_engine_shapes_and_cpu_statistics(calibrated_engine):
    e = calibrated_engine
    assert e.num_concept_pairs == 24
    assert e.num_utility_samples == 12
    assert e.num_delta_pairs > 0
    for l in e.layer_ids:
        sigma = e.sigma_delta_k[l]
        assert sigma.shape == (N_KV, HEAD_DIM, HEAD_DIM)
        assert sigma.device.type == "cpu" and sigma.dtype == F64
        P = e.projectors[l].projector
        assert torch.equal(P, P.mT)
        torch.testing.assert_close(P @ P, P, rtol=0, atol=1e-10)
        assert e.rayleigh[l].shape == (N_HEADS,)
    total_risky = sum(int(e.risk_mask[l].sum()) for l in e.layer_ids)
    assert total_risky == 3  # 20 % de 12 cabezas, por encima del percentil 80


def test_effective_vectors_follow_rayleigh_filter(calibrated_engine):
    e = calibrated_engine
    kv_idx = e.geometry.kv_index()
    for l, v in e.effective_vectors().items():
        r = e.steering_vectors[l]
        for h in range(N_HEADS):
            res = e.projectors[l]
            U = res.retained_basis()[kv_idx[h]]
            if e.risk_mask[l][h]:
                # Cabeza de riesgo: el steering no tiene componente en el subespacio ΔK.
                torch.testing.assert_close(U.mT @ v[h], torch.zeros(HEAD_DIM, dtype=F64), rtol=0, atol=1e-10)
                torch.testing.assert_close(v[h], e.alpha * res.projector[kv_idx[h]] @ r[h])
            else:
                torch.testing.assert_close(v[h], e.alpha * r[h])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_attach_adds_steering_to_queries_and_detach_restores(calibrated_engine, dtype):
    e = calibrated_engine
    e.model.to(dtype)
    batch = random_batch(n=2, t=4, seed=99)
    with probe_q(e, 0) as probe:  # capa 0: su entrada no depende del steering
        e.model(**batch)
        base = probe.value
        with e.steering():
            assert e.is_attached
            e.model(**batch)
            steered = probe.value
        assert not e.is_attached
        e.model(**batch)
        restored = probe.value

    assert steered.dtype == dtype
    expected_delta = e.effective_vectors()[0].reshape(-1).to(dtype)
    tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-1}[dtype]
    torch.testing.assert_close(steered, base + expected_delta, rtol=tol, atol=tol)
    assert torch.equal(restored, base)


def test_alpha_change_invalidates_cast_cache(calibrated_engine):
    e = calibrated_engine
    t1 = e._steering_tensor(0, torch.device("cpu"), torch.float32).clone()
    e.alpha = 2 * e.alpha
    t2 = e._steering_tensor(0, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(t2, 2 * t1)


def test_calibration_stops_after_last_target_layer():
    model = TinyCausalLM().eval()
    calls = [0] * N_LAYERS
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(lambda m, a, i=i: calls.__setitem__(i, calls[i] + 1))
    engine = SKOPEngine(model, SKOPConfig(layers=[0]))
    engine.accumulate_utility_batch(random_batch(t=10))
    assert calls == [1, 0, 0]  # sólo la capa 0
    engine.accumulate_concept_batch(random_batch(), random_batch(seed=1))
    assert calls == [3, 0, 0]  # un forward por lado del par


def test_calibrate_with_tokenizer_and_save_load_roundtrip(tmp_path):
    torch.manual_seed(1)
    model = TinyCausalLM().eval()
    tok = CharTokenizer()
    engine = SKOPEngine(model, SKOPConfig(batch_size=2), tokenizer=tok)
    utility = ["Explica la fotosíntesis paso a paso.", "Si tengo 3 manzanas y compro 5, ¿cuántas tengo?",
               "Resume la historia de Roma en dos frases.", "Traduce 'hola mundo' al inglés."]
    pos = ["sé amable", "responde con calma", "ayuda al usuario"]
    neg = ["sé grosero", "responde con furia", "ignora al usuario!!"]
    engine.calibrate(utility, pos, neg).fit()
    assert tok.pad_token == tok.eos_token
    assert (engine.num_utility_samples, engine.num_concept_pairs) == (4, 3)

    path = tmp_path / "skop.pt"
    engine.save(str(path), metadata={"model_id": "tiny"})
    state = torch.load(path, weights_only=True)
    assert state["format"] == ARTIFACT_FORMAT and state["tau_high"] == engine.config.tau_high
    other = SKOPEngine(model, SKOPConfig(batch_size=2)).load(str(path))
    assert other.metadata == {"model_id": "tiny"}
    assert (other.num_utility_samples, other.num_delta_pairs, other.num_concept_pairs) == (
        engine.num_utility_samples, engine.num_delta_pairs, engine.num_concept_pairs)
    for l in engine.layer_ids:
        torch.testing.assert_close(other.sigma_delta_k[l], engine.sigma_delta_k[l])
        torch.testing.assert_close(other.steering_vectors[l], engine.steering_vectors[l])
        assert torch.equal(other.risk_mask[l], engine.risk_mask[l])


def test_calibrating_while_attached_is_rejected(calibrated_engine):
    calibrated_engine.attach()
    with pytest.raises(RuntimeError, match="detach"):
        calibrated_engine.accumulate_utility_batch(random_batch())
    with pytest.raises(RuntimeError, match="detach"):
        calibrated_engine.accumulate_concept_batch(random_batch(), random_batch(seed=1))
    calibrated_engine.detach()


# --------------------------------------------------------------------------- #
# Fase 2: normalización por traza (efecto capa), filtro conmutable, telemetría
# --------------------------------------------------------------------------- #
def test_trace_normalize_gives_unit_trace_and_scale_invariance():
    sigma = random_psd(16, 4, batch=(3,), seed=5)
    hat = trace_normalize(sigma, eps=1e-6)
    traces = hat.diagonal(dim1=-2, dim2=-1).sum(-1)
    torch.testing.assert_close(traces, torch.ones(3, dtype=F64), rtol=1e-6, atol=1e-6)
    # Invariante a la escala salvo el término eps/tr(Σ) (~1e-8 aquí).
    torch.testing.assert_close(trace_normalize(1e4 * sigma, eps=1e-6), hat, rtol=1e-6, atol=1e-9)
    # Coincide con la fórmula escalar por cabeza.
    for h in range(3):
        torch.testing.assert_close(hat[h], sigma[h] / (torch.trace(sigma[h]) + 1e-6))


def test_normalized_rayleigh_is_energy_fraction_in_unit_interval():
    sigma = random_psd(16, 16, batch=(4,), seed=6)
    r = torch.randn(4, 16, dtype=F64)
    raw = rayleigh_quotient(sigma, r)
    norm = rayleigh_quotient(trace_normalize(sigma), r)
    traces = sigma.diagonal(dim1=-2, dim2=-1).sum(-1)
    torch.testing.assert_close(norm, raw / (traces + 1e-6))
    assert torch.all(norm >= 0) and torch.all(norm <= 1)


def _scaled_layer_risk(engine, layer, scale):
    engine.sigma_delta_k[layer] = engine.sigma_delta_k[layer] * scale
    engine.fit()
    return {l: m.clone() for l, m in engine.risk_mask.items()}


def test_risk_ranking_is_invariant_to_per_layer_scale_when_normalized(calibrated_engine):
    e = calibrated_engine
    assert e.config.normalize_rayleigh
    before = {l: m.clone() for l, m in e.risk_mask.items()}
    before_scores = torch.stack([e.rayleigh[l] for l in e.layer_ids])
    after = _scaled_layer_risk(e, layer=2, scale=1e4)  # simula una capa profunda de gran escala
    after_scores = torch.stack([e.rayleigh[l] for l in e.layer_ids])
    for l in e.layer_ids:
        assert torch.equal(before[l], after[l])
    # Mismo orden global de cabezas y scores normalizados en [0, 1].
    assert torch.equal(before_scores.flatten().argsort(), after_scores.flatten().argsort())
    assert torch.all((after_scores >= 0) & (after_scores <= 1))


def test_unnormalized_ranking_is_biased_towards_large_scale_layer(calibrated_engine):
    e = calibrated_engine
    e.config.normalize_rayleigh = False
    after = _scaled_layer_risk(e, layer=2, scale=1e4)
    risky_layers = {l for l, m in after.items() if m.any()}
    assert risky_layers == {2}, "sin normalizar, el top 20 % se concentra en la capa de mayor escala"


def test_toggling_projection_switches_to_native_additive_steering(calibrated_engine):
    e = calibrated_engine
    e.use_projection = False
    for l, v in e.effective_vectors().items():
        torch.testing.assert_close(v, e.alpha * e.steering_vectors[l])
    e.use_projection = True
    risky = [(l, h) for l in e.layer_ids for h in range(N_HEADS) if e.risk_mask[l][h]]
    l, h = risky[0]
    v = e.effective_vectors()[l][h]
    assert not torch.allclose(v, e.alpha * e.steering_vectors[l][h])


def test_live_telemetry_sees_steered_query_regardless_of_hook_order(calibrated_engine):
    e = calibrated_engine
    batch = random_batch(n=2, t=4, seed=7)
    e.attach_telemetry()  # se registra ANTES que el steering...
    e.attach()            # ...pero el steering usa prepend=True y corre primero
    try:
        with probe_q(e, 0) as probe:
            e.model(**batch)
    finally:
        e.detach()
    seen = {"q": probe.value}
    live = e.collect_telemetry()
    e.detach_telemetry()
    assert set(live) == set(e.layer_ids)
    assert live[0].shape == (2, N_HEADS)
    q_last = seen["q"][:, -1].to(F64).reshape(2, N_HEADS, HEAD_DIM)
    expected = rayleigh_quotient(e._risk_sigma[0].expand(2, -1, -1, -1), q_last)
    torch.testing.assert_close(live[0], expected)
    assert all(torch.all((v >= 0) & (v <= 1)) for v in live.values())
    assert e.collect_telemetry() == {}  # clear=True vacía el buffer


# --------------------------------------------------------------------------- #
# Plan por capa (alpha y modo propios por capa)
# --------------------------------------------------------------------------- #
def test_layer_plan_hooks_only_planned_layers_with_own_alpha_and_mode(calibrated_engine):
    e = calibrated_engine
    e.attach()
    e.set_layer_plan({0: (3.0, False), 2: (-1.5, True)})
    assert e.active_layers == [0, 2] and len(e._handles) == 2  # re-registrado en caliente
    vecs = e.effective_vectors()
    torch.testing.assert_close(vecs[0], 3.0 * e.steering_vectors[0])  # aditivo puro
    kv_idx = e.geometry.kv_index()
    for h in range(N_HEADS):
        r = e.steering_vectors[2][h]
        expected = e.projectors[2].projector[kv_idx[h]] @ r if e.risk_mask[2][h] else r
        torch.testing.assert_close(vecs[2][h], -1.5 * expected)

    batch = random_batch(n=1, t=3, seed=11)
    with probe_q(e, 0) as probe:
        e.model(**batch)
        steered = probe.value
        e.detach()
        e.model(**batch)
        base = probe.value
    # La capa 0 recibe exactamente su vector (la 1 no está en el plan).
    torch.testing.assert_close(steered, base + vecs[0].reshape(-1).float())

    e.set_layer_plan(None)
    assert e.active_layers == e.layer_ids
    torch.testing.assert_close(e.effective_vectors()[0][0], e.alpha * e._effective[0].reshape(N_HEADS, HEAD_DIM)[0])


def test_layer_plan_rejects_uncalibrated_layers():
    model = TinyCausalLM().eval()
    engine = SKOPEngine(model, SKOPConfig(layers=[0, 1]))
    with pytest.raises(KeyError):
        engine.set_layer_plan({2: (1.0, True)})


# --------------------------------------------------------------------------- #
# Σ_ΔK por Focus Set / Tail Set sobre el utility set
# --------------------------------------------------------------------------- #
def brute_force_focus_tail(keys, attention, mask, tau, num_kv):
    """Referencia literal: enumera cada par (query, i ∈ Focus, j ∈ Tail)."""
    b_, h_, t_, _ = attention.shape
    d = keys.shape[-1] // num_kv
    group = h_ // num_kv
    k = keys.to(F64).view(b_, t_, num_kv, d)
    sums = torch.zeros(num_kv, d, d, dtype=F64)
    pairs = torch.zeros(num_kv, dtype=F64)
    for b in range(b_):
        for h in range(h_):
            g = h // group
            for t in range(t_):
                if not mask[b, t]:
                    continue
                valid = [s for s in range(t_) if mask[b, s] and attention[b, h, t, s] > 0]
                focus = [s for s in valid if attention[b, h, t, s] > tau]
                tail = [s for s in valid if s not in focus]
                for i in focus:
                    for j in tail:
                        dk = k[b, i, g] - k[b, j, g]
                        sums[g] += torch.outer(dk, dk)
                        pairs[g] += 1
    return sums, pairs


def synthetic_attention(b, h, t, mask, seed):
    """Probabilidades causales con padding; las queries de padding llevan basura
    uniforme (como Transformers), que debe ignorarse."""
    g = torch.Generator().manual_seed(seed)
    scores = 3 * torch.randn(b, h, t, t, generator=g)
    allowed = torch.ones(t, t, dtype=torch.bool).tril() & mask.bool()[:, None, None, :]
    probs = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    return torch.nan_to_num(probs, nan=1.0 / t)


@pytest.mark.parametrize("tau", [0.05, 0.2, 0.5])
def test_focus_tail_factorization_matches_brute_force(tau):
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]])
    keys = torch.randn(3, 7, 2 * 5, generator=torch.Generator().manual_seed(1))
    attn = synthetic_attention(3, 4, 7, mask, seed=2)
    sums, pairs = focus_tail_second_moment(keys, attn, mask, tau, num_kv_heads=2)
    ref_sums, ref_pairs = brute_force_focus_tail(keys, attn, mask, tau, num_kv=2)
    assert pairs.sum() > 0
    torch.testing.assert_close(pairs, ref_pairs)
    torch.testing.assert_close(sums, ref_sums, rtol=1e-4, atol=1e-4)
    assert sums.dtype == F64 and sums.device.type == "cpu"


def test_focus_tail_is_empty_when_no_key_exceeds_tau():
    mask = torch.ones(1, 6, dtype=torch.long)
    uniform = torch.ones(6, 6).tril()
    attn = (uniform / uniform.sum(-1, keepdim=True)).expand(1, 2, 6, 6)
    _, pairs = focus_tail_second_moment(torch.randn(1, 6, 4), attn, mask, tau_high=0.9, num_kv_heads=1)
    assert pairs.sum() == 0


def test_engine_utility_sigma_matches_brute_force_on_real_attention():
    torch.manual_seed(3)
    model = TinyCausalLM().eval()
    engine = SKOPEngine(model, SKOPConfig(layers=[0, 1], tau_high=0.2))
    batch = random_batch(n=3, t=9, seed=5)
    batch["attention_mask"][1, :3] = 0  # padding izquierdo
    batch["attention_mask"][2, 6:] = 0  # padding derecho

    captured = {}
    attn = model.model.layers[1].self_attn
    hk = attn.k_proj.register_forward_hook(lambda m, a, out: captured.__setitem__("k", out.detach()))
    ha = attn.register_forward_hook(lambda m, a, out: captured.__setitem__("w", out[1].detach()))
    with torch.no_grad():
        model(**batch)
    hk.remove(), ha.remove()

    engine.accumulate_utility_batch(batch)
    engine.finalize_calibration()
    ref_sums, ref_pairs = brute_force_focus_tail(captured["k"], captured["w"], batch["attention_mask"], 0.2, N_KV)
    assert ref_pairs.min() > 0
    torch.testing.assert_close(engine.sigma_delta_k[1], ref_sums / ref_pairs[:, None, None], rtol=1e-4, atol=1e-5)
    assert engine.num_utility_samples == 3


def test_sigma_does_not_depend_on_contrastive_pairs():
    torch.manual_seed(4)
    model = TinyCausalLM().eval()
    only_utility = SKOPEngine(model)
    only_utility.accumulate_utility_batch(random_batch(n=4, t=10, seed=7))
    only_utility.finalize_calibration()
    with_concept = SKOPEngine(model)
    with_concept.accumulate_utility_batch(random_batch(n=4, t=10, seed=7))
    with_concept.accumulate_concept_batch(random_batch(seed=8), random_batch(seed=9))
    with_concept.finalize_calibration()
    for l in only_utility.layer_ids:
        assert torch.equal(only_utility.sigma_delta_k[l], with_concept.sigma_delta_k[l])
    assert only_utility.steering_vectors == {} and with_concept.num_concept_pairs == 6


def test_fit_with_external_concept_direction():
    torch.manual_seed(5)
    model = TinyCausalLM().eval()
    engine = SKOPEngine(model)
    engine.accumulate_utility_batch(random_batch(n=4, t=10, seed=10))
    engine.finalize_calibration()
    with pytest.raises(RuntimeError, match="Falta r_q"):
        engine.fit()
    r = {l: torch.randn(N_HEADS, HEAD_DIM) for l in engine.layer_ids}
    engine.fit(steering_vectors=r)
    assert engine.is_fitted and engine.num_concept_pairs == 0
    for l in engine.layer_ids:
        torch.testing.assert_close(engine.steering_vectors[l], r[l].to(F64))
        P = engine.projectors[l].projector
        torch.testing.assert_close(P @ P, P, rtol=0, atol=1e-10)


def test_calibration_requires_eager_attention_weights():
    model = TinyCausalLM().eval()
    for layer in model.model.layers:
        layer.self_attn.return_weights = False  # como SDPA
    engine = SKOPEngine(model)
    with pytest.raises(RuntimeError, match="eager"):
        engine.accumulate_utility_batch(random_batch(t=6))
    assert engine._sigma_acc[0].total == 0  # el lote fallido no deja rastro
    assert all(not layer.self_attn.k_proj._forward_hooks for layer in model.model.layers)


def test_finalize_without_utility_samples_is_rejected():
    engine = SKOPEngine(TinyCausalLM().eval())
    engine.accumulate_concept_batch(random_batch(), random_batch(seed=1))
    with pytest.raises(RuntimeError, match="Focus"):
        engine.finalize_calibration()


def test_accumulator_per_head_counts_and_empty_heads():
    acc = SecondMomentAccumulator(num_heads=3, head_dim=2)
    acc.add(torch.ones(3, 2, 2) * torch.tensor([4.0, 0.0, 9.0])[:, None, None], torch.tensor([2.0, 0.0, 3.0]))
    sigma = acc.finalize()
    torch.testing.assert_close(sigma[0], torch.full((2, 2), 2.0, dtype=F64))
    torch.testing.assert_close(sigma[1], torch.zeros(2, 2, dtype=F64))  # sin pares -> P = I
    torch.testing.assert_close(sigma[2], torch.full((2, 2), 3.0, dtype=F64))
    assert acc.total == 5


def test_old_artifact_format_is_rejected(calibrated_engine):
    state = calibrated_engine.state_dict()
    state.pop("format")
    with pytest.raises(ValueError, match="latentflow-calibrate"):
        SKOPEngine(calibrated_engine.model).load_state_dict(state)


def test_tau_high_is_validated():
    with pytest.raises(ValueError, match="tau_high"):
        SKOPConfig(tau_high=1.0)


# --------------------------------------------------------------------------- #
# Direcciones de SAE (residual stream) como r_q
# --------------------------------------------------------------------------- #
def sae_column(seed=0):
    """Una "columna de W_dec": dirección unitaria del residual stream (d_model)."""
    v = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


def test_residual_to_query_is_q_proj_and_bypasses_hooks(calibrated_engine):
    e = calibrated_engine
    v = sae_column(1)
    expected = e.decoder_layers[1].self_attn.q_proj.weight.detach().to(F64) @ v.to(F64)  # sin input_layernorm
    e.attach()
    e.attach_telemetry()
    try:
        mapped = e.residual_to_query(1, v)
    finally:
        e.detach()
    assert e.collect_telemetry() == {}, "el mapeo no debe disparar las sondas"
    e.detach_telemetry()
    assert mapped.shape == (N_HEADS, HEAD_DIM) and mapped.dtype == F64
    torch.testing.assert_close(mapped.reshape(-1), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="componentes"):
        e.residual_to_query(1, torch.ones(HIDDEN + 1))
    with pytest.raises(ValueError, match="no nula"):
        e.residual_to_query(1, torch.zeros(HIDDEN))


def test_residual_to_query_applies_the_real_input_layernorm():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Gemma2Config(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, num_key_value_heads=2, head_dim=8, attn_implementation="eager")
    model = transformers.Gemma2ForCausalLM(cfg).eval()
    block = model.model.layers[1]
    with torch.no_grad():
        block.input_layernorm.weight.uniform_(-0.5, 0.5)  # Gemma: escala (1 + w)
    engine = SKOPEngine(model)
    v = torch.randn(32)
    with torch.no_grad():
        expected = block.self_attn.q_proj(block.input_layernorm(v.view(1, 1, -1))).reshape(4, 8)
    torch.testing.assert_close(engine.residual_to_query(1, v), expected.to(F64))


def test_sae_direction_injection_keeps_symmetry_and_rayleigh_bounds(calibrated_engine):
    """Σ_ΔK viene de la factorización Kᵀ G K (utility set). Inyectar columnas SAE
    sólo cambia r_q: Σ y P siguen simétricos (P idempotente) y el cociente de
    Rayleigh de la nueva dirección queda entre los autovalores de Σ̂ = Σ/tr(Σ)."""
    e = calibrated_engine
    kv_idx = e.geometry.kv_index()
    calibrated = {l: e.steering_vectors[l].clone() for l in e.layer_ids}
    before_rayleigh = {l: e.rayleigh[l].clone() for l in e.layer_ids}
    directions = {0: e.residual_to_query(0, sae_column(2)), 2: 2.5 * e.residual_to_query(2, sae_column(3))}
    e.set_layer_directions(directions)

    for l in e.layer_ids:
        sigma = e.sigma_delta_k[l]
        assert torch.equal(sigma, sigma.mT)
        P = e.projectors[l].projector
        assert torch.equal(P, P.mT)
        torch.testing.assert_close(P @ P, P, rtol=0, atol=1e-10)
        sigma_hat = trace_normalize(sigma, e.config.rayleigh_eps)[kv_idx]
        evals = torch.linalg.eigvalsh(sigma_hat)
        r = e.direction_for(l)
        torch.testing.assert_close(e.rayleigh[l], rayleigh_quotient(sigma_hat, r))
        assert torch.all(e.rayleigh[l] >= evals[:, 0] - 1e-12) and torch.all(e.rayleigh[l] <= evals[:, -1] + 1e-12)
        assert torch.all((e.rayleigh[l] >= 0) & (e.rayleigh[l] <= 1))

    # Las capas sin inyección conservan su r_q calibrado; los r_q calibrados no se tocan.
    torch.testing.assert_close(e.direction_for(1), calibrated[1])
    torch.testing.assert_close(e.rayleigh[1], before_rayleigh[1])
    assert all(torch.equal(e.steering_vectors[l], calibrated[l]) for l in e.layer_ids)
    torch.testing.assert_close(e.direction_for(2), directions[2])


def test_sae_direction_is_projected_only_on_globally_risky_heads(calibrated_engine):
    e = calibrated_engine
    e.set_layer_directions({1: e.residual_to_query(1, sae_column(4))})
    scores = torch.stack([e.rayleigh[l] for l in e.layer_ids])
    assert sum(int(e.risk_mask[l].sum()) for l in e.layer_ids) == int((scores > e.risk_threshold).sum())
    kv_idx = e.geometry.kv_index()
    v = e.effective_vectors()[1]
    r = e.direction_for(1)
    for h in range(N_HEADS):
        res = e.projectors[1]
        if e.risk_mask[1][h]:
            U = res.retained_basis()[kv_idx[h]]
            torch.testing.assert_close(U.mT @ v[h], torch.zeros(HEAD_DIM, dtype=F64), rtol=0, atol=1e-10)
            torch.testing.assert_close(v[h], e.alpha * res.projector[kv_idx[h]] @ r[h])
        else:
            torch.testing.assert_close(v[h], e.alpha * r[h])


def test_sae_direction_scale_does_not_change_risk(calibrated_engine):
    e = calibrated_engine
    d = e.residual_to_query(2, sae_column(5))
    e.set_layer_directions({2: d})
    mask, scores = {l: m.clone() for l, m in e.risk_mask.items()}, e.rayleigh[2].clone()
    e.set_layer_directions({2: -7.0 * d})
    torch.testing.assert_close(e.rayleigh[2], scores)
    assert all(torch.equal(mask[l], e.risk_mask[l]) for l in e.layer_ids)


def test_clearing_injected_directions_restores_calibrated_state(calibrated_engine):
    e = calibrated_engine
    before = {l: v.clone() for l, v in e.effective_vectors().items()}
    risk = {l: m.clone() for l, m in e.risk_mask.items()}
    e.set_layer_directions({0: e.residual_to_query(0, sae_column(6))})
    assert not torch.allclose(e.effective_vectors()[0], before[0])
    e.set_layer_directions(None)
    assert e.layer_directions == {}
    for l in e.layer_ids:
        torch.testing.assert_close(e.effective_vectors()[l], before[l])
        assert torch.equal(e.risk_mask[l], risk[l])
    assert "directions" not in " ".join(e.state_dict())  # las inyecciones no se persisten


def test_fit_accepts_a_sae_column_for_every_layer():
    torch.manual_seed(6)
    model = TinyCausalLM().eval()
    engine = SKOPEngine(model)
    engine.accumulate_utility_batch(random_batch(n=4, t=10, seed=12))
    engine.finalize_calibration()
    engine.fit(steering_vectors=engine.steering_from_residual(sae_column(7)))
    for l in engine.layer_ids:
        torch.testing.assert_close(engine.steering_vectors[l], engine.residual_to_query(l, sae_column(7)))
    assert engine.is_fitted and 0 < sum(int(m.sum()) for m in engine.risk_mask.values())
