"""Cortex API — FastAPI server exposing search, ask, and stats endpoints."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph.store import BrainStore
from query.engine import QueryEngine

# Module-level placeholders — populated in lifespan so importing this module
# does NOT open the Kuzu database (avoids lock conflicts in tests).
store: BrainStore = None  # type: ignore[assignment]
engine: QueryEngine = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, engine
    store = BrainStore()
    engine = QueryEngine(store)
    yield
    # nothing to clean up — Kuzu/ChromaDB close on GC


app = FastAPI(title="Cortex", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    model: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/ask")
def ask(req: AskRequest):
    answer, model_used = engine.ask_with_model(req.question, model=req.model)
    sources = engine.search(req.question, top_k=3)
    return {
        "answer": answer,
        "model_used": model_used,
        "sources": [
            {"title": s["title"], "path": s["path"], "tags": s.get("tags", "")}
            for s in sources
        ],
    }


@app.post("/api/search")
def search(req: SearchRequest):
    results = engine.search(req.query, top_k=req.top_k)
    return {
        "results": [
            {
                "title": r["title"],
                "path": r["path"],
                "tags": r.get("tags", ""),
                "snippet": r["content"][:200],
            }
            for r in results
        ]
    }


@app.get("/api/stats")
def stats():
    counts = {}
    for label, query in [
        ("notes", "MATCH (n:Note) RETURN count(n)"),
        ("tags", "MATCH (t:Tag) RETURN count(t)"),
        ("entities", "MATCH (e:Entity) RETURN count(e)"),
        ("tag_links", "MATCH ()-[r:TAGGED]->() RETURN count(r)"),
        ("entity_links", "MATCH ()-[r:LINKS_TO]->() RETURN count(r)"),
    ]:
        try:
            result = store.conn.execute(query)
            counts[label] = result.get_next()[0]
        except Exception:
            counts[label] = 0
    return counts


@app.get("/api/graph/entities")
def list_entities():
    """Return all entities in the graph for visualization."""
    try:
        result = store.conn.execute(
            "MATCH (e:Entity) RETURN e.name ORDER BY e.name"
        )
        return {"entities": [row[0] for row in result.get_as_df().itertuples(index=False)]}
    except Exception:
        return {"entities": []}


@app.get("/api/graph/tags")
def list_tags():
    """Return all tags with note counts."""
    try:
        result = store.conn.execute("""
            MATCH (t:Tag)<-[:TAGGED]-(n:Note)
            RETURN t.name, count(n) AS cnt
            ORDER BY cnt DESC
        """)
        return {
            "tags": [
                {"name": row[0], "count": row[1]}
                for row in result.get_as_df().itertuples(index=False)
            ]
        }
    except Exception:
        return {"tags": []}


# ── Code endpoints ───────────────────────────────────────────────────


class SearchCodeRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/search-code")
def search_code(req: SearchCodeRequest):
    """Search only indexed code (files, functions, classes)."""
    results = engine.search_code(req.query, top_k=req.top_k)
    return {
        "results": [
            {
                "title": r["title"],
                "path": r["path"],
                "type": r.get("type", "code_file"),
                "language": r.get("language", ""),
                "name": r.get("name", ""),
                "snippet": r["content"][:200],
            }
            for r in results
        ]
    }


@app.get("/api/code-stats")
def code_stats():
    """Return counts of indexed code nodes."""
    return store.get_code_stats()
