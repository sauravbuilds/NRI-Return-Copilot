"""Search the official-document index. Higher score always means more relevant."""
from dataclasses import dataclass

import config

_store = None


@dataclass
class Chunk:
    text: str
    metadata: dict
    score: float  # 0..1-ish, higher is better


def _get_store():
    global _store
    if _store is None:
        from langchain_chroma import Chroma
        from scripts.ingest import get_embeddings
        _store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(config.CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _store


def retrieve(query: str, k: int = config.TOP_K) -> list[Chunk]:
    # With a cosine index LangChain converts distance to relevance (1 - distance),
    # so higher is better. Checked in scripts/score_report.py output.
    results = _get_store().similarity_search_with_relevance_scores(query, k=k)
    return [Chunk(doc.page_content, doc.metadata, float(score)) for doc, score in results]
