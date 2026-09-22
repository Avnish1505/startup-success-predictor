from src.advisor.chunking import chunk_markdown_document

SHORT_DOC = """---
title: Test Doc
source_url: https://example.com/test
---

## Section One

This is a short section with only a few words in it.

## Section Two

This is another short section, also brief.
"""

_LONG_SENTENCES = " ".join(f"This is sentence number {i} in a long section about startups." for i in range(80))
LONG_SECTION_DOC = f"""---
title: Long Doc
source_url: https://example.com/long
---

## A Long Section

{_LONG_SENTENCES}
"""


def test_short_sections_become_one_chunk_each():
    chunks = chunk_markdown_document(SHORT_DOC, doc_id="test_doc", source_url="https://example.com/test")
    assert len(chunks) == 2
    assert chunks[0].heading == "Section One"
    assert chunks[1].heading == "Section Two"
    assert all(c.source_url == "https://example.com/test" for c in chunks)
    assert all(c.doc_id == "test_doc" for c in chunks)


def test_chunk_ids_are_unique_and_stable():
    chunks = chunk_markdown_document(SHORT_DOC, doc_id="test_doc", source_url="https://example.com/test")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("test_doc") for cid in ids)


def test_long_section_splits_with_overlap():
    chunks = chunk_markdown_document(
        LONG_SECTION_DOC, doc_id="long_doc", source_url="https://example.com/long",
        target_words=200, overlap_words=30, max_section_words=400,
    )
    assert len(chunks) > 1
    for c in chunks:
        word_count = len(c.text.split())
        assert word_count <= 200 + 30 + 20  # target + overlap + slack for sentence-boundary rounding
    # verify actual overlap: some words from the end of chunk N appear at the start of chunk N+1
    first_words = set(chunks[0].text.split()[-30:])
    second_words = set(chunks[1].text.split()[:30])
    assert first_words & second_words


def test_never_splits_a_short_section_that_fits():
    chunks = chunk_markdown_document(SHORT_DOC, doc_id="test_doc", source_url="https://example.com/test")
    assert "This is a short section with only a few words in it." in chunks[0].text
