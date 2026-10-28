"""Resolución de features de SAE (Gemma Scope) sin red."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from latentflow.core.sae import (
    FeatureError,
    FeatureNotFoundError,
    FeatureRef,
    GemmaScopeResolver,
    HashFeatureResolver,
)

ROOT = Path(__file__).resolve().parents[1]
FID = "gemma-2-2b/20-gemmascope-res-16k/7"
D_MODEL, D_SAE = 6, 10


def w_dec(seed=0):
    return torch.randn(D_SAE, D_MODEL, generator=torch.Generator().manual_seed(seed))  # (d_sae, d_model), como Gemma Scope


def save_npz(path: Path, matrix: torch.Tensor) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W_dec=matrix.numpy(), W_enc=matrix.T.numpy(), b_dec=np.zeros(D_MODEL, np.float32))
    return path


def test_parse_neuronpedia_ids():
    ref = FeatureRef.parse(" gemma-2-2b/20-gemmascope-res-16k/12082 ")
    assert (ref.model, ref.sae_id, ref.index, ref.layer, ref.site, ref.width, ref.model_size) == (
        "gemma-2-2b", "20-gemmascope-res-16k", 12082, 20, "res", "16k", "2b")
    generic = FeatureRef.parse("my-model/custom_sae.v2/5")
    assert (generic.sae_id, generic.index, generic.layer) == ("custom_sae.v2", 5, None)
    for bad in ("12082", "a/b", "a/b/c", ""):
        with pytest.raises(FeatureError):
            FeatureRef.parse(bad)


def test_hash_resolver_is_deterministic_across_processes():
    resolver = HashFeatureResolver()
    a = resolver.resolve(FID, 32)
    assert a.source == "dev-hash" and (a.layer, a.site) == (20, "res")
    assert torch.allclose(a.vector.norm(), torch.tensor(1.0))
    torch.testing.assert_close(resolver.resolve(FID, 32).vector, a.vector)
    assert not torch.allclose(resolver.resolve(FID + "1", 32).vector, a.vector)
    # Otro proceso (otra semilla de hash() de Python) produce el mismo vector.
    code = f"from latentflow.core.sae import HashFeatureResolver as H; print(H().resolve({FID!r}, 32).vector.tolist())"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120, check=True)
    torch.testing.assert_close(torch.tensor(eval(out.stdout)), a.vector)
    with pytest.raises(FeatureError):
        resolver.resolve("  ", 32)


def test_gemma_scope_tree_npz(tmp_path):
    matrix = w_dec()
    save_npz(tmp_path / "gemma-scope-2b-pt-res-canonical" / "layer_20" / "width_16k" / "canonical" / "params.npz", matrix)
    resolved = GemmaScopeResolver(str(tmp_path), use_hf_cache=False).resolve(FID, D_MODEL)
    torch.testing.assert_close(resolved.vector, matrix[7])
    assert resolved.source.startswith("gemma-scope:") and resolved.layer == 20


def test_single_average_l0_variant_is_used(tmp_path):
    matrix = w_dec(1)
    save_npz(tmp_path / "layer_20" / "width_16k" / "average_l0_71" / "params.npz", matrix)
    torch.testing.assert_close(GemmaScopeResolver(str(tmp_path), use_hf_cache=False).resolve(FID, D_MODEL).vector, matrix[7])


def test_structured_files_in_both_orientations(tmp_path):
    from safetensors.torch import save_file

    matrix = w_dec(2)
    base = tmp_path / "gemma-2-2b"
    base.mkdir()
    save_file({"W_dec": matrix.T.contiguous()}, str(base / "20-gemmascope-res-16k.safetensors"))  # (d_model, d_sae)
    torch.testing.assert_close(GemmaScopeResolver(str(tmp_path), use_hf_cache=False).resolve(FID, D_MODEL).vector, matrix[7])

    torch.save({"W_dec": matrix}, base / "custom.pt")
    torch.testing.assert_close(
        GemmaScopeResolver(str(tmp_path), use_hf_cache=False).resolve("gemma-2-2b/custom/3", D_MODEL).vector, matrix[3])


def test_local_hugging_face_cache_without_network(tmp_path):
    repo = tmp_path / "models--google--gemma-scope-2b-pt-res-canonical"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123")
    matrix = w_dec(3)
    save_npz(repo / "snapshots" / "abc123" / "layer_20" / "width_16k" / "canonical" / "params.npz", matrix)
    resolver = GemmaScopeResolver(None, hf_cache_dir=str(tmp_path))
    torch.testing.assert_close(resolver.resolve(FID, D_MODEL).vector, matrix[7])


def test_resolution_errors(tmp_path):
    resolver = GemmaScopeResolver(str(tmp_path), hf_cache_dir=str(tmp_path / "hf"))
    with pytest.raises(FeatureNotFoundError, match="huggingface-cli download google/gemma-scope-2b-pt-res-canonical"):
        resolver.resolve(FID, D_MODEL)
    with pytest.raises(FeatureError, match="atención"):
        resolver.resolve("gemma-2-2b/20-gemmascope-att-16k/7", D_MODEL)
    save_npz(tmp_path / "layer_20" / "width_16k" / "canonical" / "params.npz", w_dec())
    with pytest.raises(FeatureError, match="d_model=2304"):
        resolver.resolve(FID, 2304)
    with pytest.raises(FeatureError, match="fuera de rango"):
        resolver.resolve("gemma-2-2b/20-gemmascope-res-16k/99", D_MODEL)


def test_resolved_vectors_are_cached(tmp_path):
    path = save_npz(tmp_path / "layer_20" / "width_16k" / "canonical" / "params.npz", w_dec())
    resolver = GemmaScopeResolver(str(tmp_path), use_hf_cache=False)
    first = resolver.resolve(FID, D_MODEL)
    path.unlink()
    assert resolver.resolve(FID, D_MODEL) is first
