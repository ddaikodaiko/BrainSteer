"""Dependencias declaradas en ``pyproject.toml`` (#12 y #13 de la auditoría)."""

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
from packaging.requirements import Requirement  # noqa: E402
from packaging.version import Version  # noqa: E402

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def extra(name):
    reqs = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"][name]
    return {r.name: r for r in map(Requirement, reqs)}


@pytest.mark.parametrize("name", ["server", "models"])
def test_transformers_floor_returns_eager_attention_weights(name):
    # Antes de 4.48 la atención eager devolvía pesos None sin `output_attentions`.
    spec = extra(name)["transformers"].specifier
    assert not spec.contains(Version("4.47.1")) and spec.contains(Version("4.48.0"))


def test_server_extra_alone_can_load_with_device_map_auto():
    assert "accelerate" in extra("server")
