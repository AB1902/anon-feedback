"""Vector retrieval over (a) HR handbook clauses and (b) anonymised feedback.

Primary backend: ChromaDB with its default sentence-embedding model (all-MiniLM-L6-v2,
run locally via ONNX; downloaded once on first use, ~80 MB).
Fallback: BM25 keyword search, used automatically if ChromaDB or the model is unavailable,
so the app never breaks on a machine without internet.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from app.config import CHROMA_DIR, DATA_DIR

HANDBOOK = DATA_DIR / "handbook" / "handbook.md"


@dataclass
class Doc:
    id: str
    text: str
    meta: dict


# ------------------------------------------------------------------ handbook chunks

@lru_cache(maxsize=1)
def handbook_chunks() -> list[Doc]:
    """One chunk per numbered clause ("### 4.2 Earned / privilege leave")."""
    md = HANDBOOK.read_text(encoding="utf-8")
    docs, section = [], ""
    for block in re.split(r"\n(?=#{2,3} )", md):
        head = block.split("\n", 1)[0]
        if head.startswith("## "):
            section = head[3:].strip()
            continue
        m = re.match(r"### (\d+\.\d+)\s+(.*)", head)
        if m:
            body = block.split("\n", 1)[1].strip() if "\n" in block else ""
            body = re.sub(r"\n-{3,}\s*$", "", body).strip()
            docs.append(Doc(m.group(1), f"{section} > {m.group(2)}\n{body}",
                            {"clause": m.group(1), "title": m.group(2), "section": section}))
    return docs


# ------------------------------------------------------------------ BM25 fallback

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the and or of to in on for is are was were be been it this that with as at by "
            "from my me i we our they their he she his her you your not no do does did have has had "
            "but so if than then there here about into over under very just also can could would should".split())


def _tok(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if w not in _STOP and len(w) > 1]


class BM25:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.toks = [_tok(d.text) for d in docs]
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def search(self, query: str, k: int = 3, where=None) -> list[tuple[Doc, float]]:
        q = _tok(query)
        scored = []
        for i, d in enumerate(self.docs):
            if where and any(d.meta.get(key) != val for key, val in where.items()):
                continue
            s = 0.0
            for w in q:
                f = self.tf[i].get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(self.toks[i]) / self.avg))
            if s > 0:
                scored.append((d, s))
        return sorted(scored, key=lambda x: -x[1])[:k]


# ------------------------------------------------------------------ Chroma

@lru_cache(maxsize=1)
def _client():
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _collection(name: str):
    from chromadb.utils import embedding_functions
    return _client().get_or_create_collection(
        name, embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"})


class Index:
    """Uses Chroma when it works, otherwise BM25. `backend` says which one is active."""

    def __init__(self, name: str, docs: list[Doc]):
        self.name, self.docs, self.backend = name, docs, "bm25"
        self._bm25 = BM25(docs)
        try:
            col = _collection(name)
            if col.count() != len(docs):
                existing = col.get()["ids"]
                if existing:
                    col.delete(ids=existing)
                for i in range(0, len(docs), 200):
                    part = docs[i:i + 200]
                    col.add(ids=[d.id for d in part], documents=[d.text for d in part],
                            metadatas=[{k: (v if v is not None else "") for k, v in d.meta.items()} for d in part])
            self._col = col
            self.backend = "chromadb"
        except Exception as e:
            print(f"[retrieval] ChromaDB unavailable for '{name}', using BM25: {str(e)[:120]}")

    def search(self, query: str, k: int = 3, where: dict | None = None) -> list[Doc]:
        if self.backend == "chromadb":
            try:
                res = self._col.query(query_texts=[query], n_results=k, where=where or None)
                by_id = {d.id: d for d in self.docs}
                return [by_id[i] for i in res["ids"][0] if i in by_id]
            except Exception as e:
                print(f"[retrieval] Chroma query failed, BM25 instead: {str(e)[:120]}")
        return [d for d, _ in self._bm25.search(query, k, where)]


@lru_cache(maxsize=1)
def handbook_index() -> Index:
    return Index("handbook", handbook_chunks())


def retrieve_policy(query: str, k: int = 3) -> list[Doc]:
    return handbook_index().search(query, k)
