from typing import List
import logfire


def _split_oversized_paragraph(paragraph: str, chunk_size: int) -> List[str]:
    """Split a long paragraph on whitespace, hard-splitting oversized tokens."""
    pieces: List[str] = []
    current_piece = ""

    for word in paragraph.split():
        while len(word) > chunk_size:
            if current_piece:
                pieces.append(current_piece)
                current_piece = ""
            pieces.append(word[:chunk_size])
            word = word[chunk_size:]

        candidate = f"{current_piece} {word}".strip()
        if len(candidate) > chunk_size:
            if current_piece:
                pieces.append(current_piece)
            current_piece = word
        else:
            current_piece = candidate

    if current_piece:
        pieces.append(current_piece)
    return pieces


def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Split text by paragraphs, then by words when a paragraph exceeds the limit.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.extend(_split_oversized_paragraph(paragraph, chunk_size))
                continue

            candidate = f"{current_chunk}\n\n{paragraph}".strip()
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks