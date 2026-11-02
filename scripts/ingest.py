"""Build the ChromaDB index from data/official/manifest.csv.

Run:  python scripts/ingest.py
It rebuilds from scratch every time, so the index always matches the PDFs.

Each PDF page becomes its own document first, so a chunk can never lose
its page number. Every chunk carries the metadata a citation needs.
"""
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def load_manifest(path=config.MANIFEST_PATH) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pdf_to_chunks(row: dict, official_dir=config.OFFICIAL_DIR,
                  chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages = PyPDFLoader(str(Path(official_dir) / f"{row['doc_id']}.pdf")).load()
    for i, p in enumerate(pages, start=1):
        # Check the recorded page number against the real one (0-based in the loader).
        assert p.metadata["page"] + 1 == i, f"page mismatch in {row['doc_id']}"
        p.metadata = {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "issuer": row["issuer"],
            "url": row["url"],
            "page": i,  # 1-based, matches the PDF viewer
            "page_url": f"{row['url']}#page={i}",
            "date_published": row.get("date_published", ""),
            "last_verified": row.get("last_verified", ""),
            "scenario_tags": row.get("scenario_tags", ""),
        }
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)  # splitter copies metadata to every chunk


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def build_index(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP) -> int:
    from langchain_chroma import Chroma

    chunks = []
    for row in load_manifest():
        if not (config.OFFICIAL_DIR / f"{row['doc_id']}.pdf").exists():
            print(f"SKIP {row['doc_id']}: file missing")
            continue
        chunks += pdf_to_chunks(row, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)  # clean rebuild
    Chroma.from_documents(
        chunks, get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Indexed {len(chunks)} chunks")
    return len(chunks)


if __name__ == "__main__":
    build_index()
