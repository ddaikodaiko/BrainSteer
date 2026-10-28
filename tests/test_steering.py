from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latentflow.core.steering import (
    SecondMomentAccumulator,
    SKOPConfig,
    SKOPEngine,
    _aggregate_tokens,
    compute_orthogonal_projector,
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


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, N_KV * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        return self.o_proj(q) + k.mean(-1, keepdim=True)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()

    def forward(self, x):
        return x + self.self_attn(x)


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
            h = layer(h)
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


@pytest.fixture
def calibrated_engine():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    engine = SKOPEngine(model, SKOPConfig(alpha=2.0, gamma_energy=0.90, risk_percentile=0.80))
    for i in range(4):
        engine.accumulate_pair_batch(random_batch(seed=2 * i), random_batch(seed=2 * i + 1))
    return engine.finalize_calibration().fit()


def test_engine_shapes_and_cpu_statistics(calibrated_engine):
    e = calibrated_engine
    assert e.num_calibration_pairs == 24
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
    # La entrada de o_proj es la q ya steerada (el hook de q_proj se aplica antes).
    o_proj = e.decoder_layers[0].self_attn.o_proj  # capa 0: su entrada no depende del steering
    seen = {}
    probe = o_proj.register_forward_pre_hook(lambda m, args: seen.__setitem__("q", args[0]))
    try:
        e.model(**batch)
        base = seen["q"].clone()
        with e.steering():
            assert e.is_attached
            e.model(**batch)
            steered = seen["q"].clone()
        assert not e.is_attached
        e.model(**batch)
        restored = seen["q"].clone()
    finally:
        probe.remove()

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
    engine.accumulate_pair_batch(random_batch(), random_batch(seed=1))
    assert calls == [2, 0, 0]  # sólo la capa 0, una vez por forward (positivo y negativo)


def test_calibrate_with_tokenizer_and_save_load_roundtrip(tmp_path):
    torch.manual_seed(1)
    model = TinyCausalLM().eval()
    tok = CharTokenizer()
    engine = SKOPEngine(model, SKOPConfig(batch_size=2), tokenizer=tok)
    pos = ["sé amable", "responde con calma", "ayuda al usuario"]
    neg = ["sé grosero", "responde con furia", "ignora al usuario!!"]
    engine.calibrate(pos, neg).fit()
    assert tok.pad_token == tok.eos_token
    assert engine.num_calibration_pairs == 3

    path = tmp_path / "skop.pt"
    engine.save(str(path))
    other = SKOPEngine(model, SKOPConfig(batch_size=2)).load(str(path))
    for l in engine.layer_ids:
        torch.testing.assert_close(other.sigma_delta_k[l], engine.sigma_delta_k[l])
        assert torch.equal(other.risk_mask[l], engine.risk_mask[l])


def test_calibrating_while_attached_is_rejected(calibrated_engine):
    calibrated_engine.attach()
    with pytest.raises(RuntimeError, match="detach"):
        calibrated_engine.accumulate_pair_batch(random_batch(), random_batch(seed=1))
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
    seen = {}
    probe = e.decoder_layers[0].self_attn.o_proj.register_forward_pre_hook(
        lambda m, args: seen.__setitem__("q", args[0])
    )
    try:
        e.model(**batch)
    finally:
        probe.remove()
        e.detach()
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
    seen = {}
    hooks = [
        e.decoder_layers[l].self_attn.o_proj.register_forward_pre_hook(
            lambda m, a, l=l: seen.__setitem__(l, a[0].clone())
        )
        for l in (0, 1)
    ]
    try:
        e.model(**batch)
        steered = dict(seen)
        e.detach()
        e.model(**batch)
    finally:
        for h in hooks:
            h.remove()
    # La capa 0 recibe exactamente su vector; la 1 no está en el plan.
    torch.testing.assert_close(steered[0], seen[0] + vecs[0].reshape(-1).float())

    e.set_layer_plan(None)
    assert e.active_layers == e.layer_ids
    torch.testing.assert_close(e.effective_vectors()[0][0], e.alpha * e._effective[0].reshape(N_HEADS, HEAD_DIM)[0])


def test_layer_plan_rejects_uncalibrated_layers():
    model = TinyCausalLM().eval()
    engine = SKOPEngine(model, SKOPConfig(layers=[0, 1]))
    with pytest.raises(KeyError):
        engine.set_layer_plan({2: (1.0, True)})
