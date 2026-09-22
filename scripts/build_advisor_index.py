"""Build the real advisor retrieval index from data/corpus/*.md and save it
to models/production/advisor_index/. Run this whenever the corpus changes."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import frontmatter  # noqa: E402

from src.advisor.chunking import chunk_markdown_document  # noqa: E402
from src.advisor.retrieval import build_index, save_index  # noqa: E402

CORPUS_DIR = REPO_ROOT / "data" / "corpus"
OUT_DIR = REPO_ROOT / "models" / "production" / "advisor_index"


def main() -> None:
    all_chunks = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        post = frontmatter.load(path)
        doc_id = path.stem
        source_url = post.metadata["source_url"]
        chunks = chunk_markdown_document(path.read_text(), doc_id=doc_id, source_url=source_url)
        all_chunks.extend(chunks)
        print(f"{path.name}: {len(chunks)} chunks")

    index = build_index(all_chunks)
    save_index(index, OUT_DIR)
    print(f"Wrote {len(all_chunks)} total chunks to {OUT_DIR}/ (vectors shape {index.vectors.shape})")


if __name__ == "__main__":
    main()
