"""Markdown chunker for the advisor's retrieval corpus. See the strategy
note in the implementation plan / RETRIEVAL_EVAL.md for why these numbers
were chosen - target ~200 words, 30-word overlap, never split a section
that fits, never split mid-sentence."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    heading: str
    text: str
    source_url: str


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\n")
    return text


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on ## headings. Returns [(heading, body), ...]."""
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))
    return sections


def _split_long_section(body: str, target_words: int, overlap_words: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", body)
    if len(sentences) == 1 and len(sentences[0].split()) > target_words * 2:
        # No sentence punctuation to split on (e.g. a run-on section with no
        # periods) - fall back to hard word-count splitting rather than
        # emitting one oversized chunk.
        words = sentences[0].split()
        sentences = [
            " ".join(words[i : i + target_words])
            for i in range(0, len(words), max(target_words - overlap_words, 1))
        ]
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        current.append(sentence)
        current_words += len(sentence_words)
        if current_words >= target_words:
            pieces.append(" ".join(current))
            overlap_tail = " ".join(current).split()[-overlap_words:]
            current = [" ".join(overlap_tail)]
            current_words = len(overlap_tail)
    if current and " ".join(current).strip():
        pieces.append(" ".join(current))
    return pieces


def chunk_markdown_document(
    text: str,
    doc_id: str,
    source_url: str,
    target_words: int = 200,
    overlap_words: int = 30,
    max_section_words: int = 400,
) -> list[Chunk]:
    body = _strip_frontmatter(text)
    chunks: list[Chunk] = []
    for heading, section_body in _split_sections(body):
        word_count = len(section_body.split())
        if word_count <= max_section_words:
            pieces = [section_body]
        else:
            pieces = _split_long_section(section_body, target_words, overlap_words)
        for i, piece in enumerate(pieces):
            suffix = f"_{i}" if len(pieces) > 1 else ""
            chunk_id = f"{doc_id}::{heading.lower().replace(' ', '_')}{suffix}"
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, heading=heading, text=piece, source_url=source_url))
    return chunks
