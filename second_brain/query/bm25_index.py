"""BM25 keyword index over the ChromaDB collection.

Built in-memory on startup from documents already stored in ChromaDB —
no extra persistence needed. For a vault of ~500–2000 notes, build time
is under 1s and memory use is negligible.
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self) -> None:
        self._corpus: list[dict] = []  # [{title, path, content, tags, ...}]
        self._bm25: BM25Okapi | None = None

    def build(self, collection) -> None:
        """Pull all documents from a ChromaDB collection and build the index."""
        data = collection.get(include=["documents", "metadatas"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        self._corpus = [
            {**m, "content": d}
            for m, d in zip(metas, docs)
        ]
        if not self._corpus:
            self._bm25 = None
            return
        tokenized = [doc["content"].lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return up to top_k documents ranked by BM25 score."""
        if not self._bm25 or not self._corpus:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            self._corpus[i]
            for i in ranked[:top_k]
            if scores[i] > 0
        ]
