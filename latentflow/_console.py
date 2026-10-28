"""Salida de consola de los CLIs de LatentFlow."""

from __future__ import annotations

import sys

__all__ = ["ensure_utf8_stdio"]


def ensure_utf8_stdio() -> None:
    """Pasa ``stdout`` y ``stderr`` a UTF-8 si usan otra codificación.

    Los CLIs imprimen Σ, Δ o τ. En Windows, con la salida redirigida a un fichero
    o a otro proceso, Python usa la página de códigos del sistema (cp1252), que no
    los tiene, y la escritura fallaría con ``UnicodeEncodeError``.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
