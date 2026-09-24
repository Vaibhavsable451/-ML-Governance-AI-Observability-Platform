from __future__ import annotations


def chunk_text(text: str, source: str, max_chars: int = 500) -> list[dict]:
    """Paragraph-aware chunking: merge paragraphs until max_chars, keep the section heading as context."""
    chunks, buf, heading = [], "", ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if para.startswith("#"):
            heading = para.lstrip("# ").strip()
            continue
        candidate = f"{buf}\n{para}".strip() if buf else para
        if len(candidate) > max_chars and buf:
            chunks.append({"source": source, "text": f"{heading}: {buf}" if heading else buf})
            buf = para
        else:
            buf = candidate
    if buf:
        chunks.append({"source": source, "text": f"{heading}: {buf}" if heading else buf})
    return chunks
