from __future__ import annotations

from pathlib import Path


def load_documents(directory: Path) -> list[dict]:
    docs = []
    for p in sorted(Path(directory).glob("*.md")):
        docs.append({"source": p.stem, "text": p.read_text(encoding="utf-8")})
    return docs
