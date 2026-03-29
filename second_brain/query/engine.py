import os
import re
from typing import Any

import ollama
from graph.store import BrainStore
from query.bm25_index import BM25Index


# Questions that don't need RAG context
GREETING_PATTERNS = [
    r"^(hi|hello|hey|sup|yo|hola|salut|salam)[\s!?.]*$",
    r"^(who are you|what are you|what is this|what do you do)[\s?]*$",
    r"^(help|how does this work)[\s?]*$",
]

IDENTITY_RESPONSE = (
    "I'm Cortex — your local AI second brain. I index your Obsidian vault "
    "into a knowledge graph and let you query it with natural language. "
    "Everything runs locally on your machine. No cloud, no API keys.\n\n"
    "Try asking me things like:\n"
    "- \"What projects am I working on?\"\n"
    "- \"Find everything about ADINO\"\n"
    "- \"What ideas do I have for side projects?\"\n"
    "- \"Summarize my thesis work\""
)

GREETING_RESPONSE = (
    "Hey! I'm Cortex, your local knowledge assistant. "
    "Ask me anything about your vault — projects, notes, ideas, whatever you've written down."
)

DEFAULT_CHAT_MODEL = os.getenv("CORTEX_CHAT_MODEL", "qwen2.5:7b")
FALLBACK_CHAT_MODEL = "llama3.2:latest"
RULE_BASED_MODEL = "rule-based"


class QueryEngine:
    def __init__(self, store: BrainStore, model: str | None = None):
        self.store = store
        self.model = model or DEFAULT_CHAT_MODEL
        self.embed_model = "nomic-embed-text"
        self._resolved_model_cache: dict[str, str] = {}
        self._bm25 = BM25Index()
        self._bm25.build(store.collection)

    def embed(self, text: str) -> list[float]:
        response = ollama.embeddings(model=self.embed_model, prompt=text)
        return response["embedding"]

    def _is_embedding_model_entry(self, model_entry: Any) -> bool:
        model_name = (getattr(model_entry, "model", "") or "").lower()
        if "embed" in model_name:
            return True

        details = getattr(model_entry, "details", None)
        family = (getattr(details, "family", "") or "").lower()
        if "embed" in family or "bert" in family:
            return True

        families = getattr(details, "families", []) or []
        return any(
            "embed" in str(f).lower() or "bert" in str(f).lower()
            for f in families
        )

    def _list_installed_chat_models(self) -> list[str]:
        response = ollama.list()
        models = getattr(response, "models", []) or []
        names = [
            m.model
            for m in models
            if getattr(m, "model", None) and not self._is_embedding_model_entry(m)
        ]
        return sorted(set(names))

    def _candidate_models(
        self, preferred_model: str, installed_chat_models: list[str], exclude: set[str]
    ) -> list[str]:
        candidates: list[str] = []
        for candidate in [preferred_model, FALLBACK_CHAT_MODEL]:
            if candidate and candidate not in candidates and candidate not in exclude:
                candidates.append(candidate)

        first_local = next((m for m in installed_chat_models if m not in exclude), None)
        if first_local and first_local not in candidates:
            candidates.append(first_local)
        return candidates

    def _resolve_chat_model(
        self,
        preferred_model: str,
        force_refresh: bool = False,
        exclude: set[str] | None = None,
    ) -> str:
        excluded = exclude or set()
        if not force_refresh and not excluded and preferred_model in self._resolved_model_cache:
            return self._resolved_model_cache[preferred_model]

        installed_chat_models = self._list_installed_chat_models()
        installed_lookup = set(installed_chat_models)
        candidates = self._candidate_models(preferred_model, installed_chat_models, excluded)

        for candidate in candidates:
            if candidate in installed_lookup:
                if not excluded:
                    self._resolved_model_cache[preferred_model] = candidate
                return candidate

        raise RuntimeError(
            "No usable local chat model found. Pull a model with `ollama pull qwen2.5:7b`."
        )

    def _is_model_not_found_error(self, error: Exception) -> bool:
        if not isinstance(error, ollama.ResponseError):
            return False
        if getattr(error, "status_code", None) == 404:
            return True
        error_text = str(getattr(error, "error", "") or error).lower()
        return "model" in error_text and "not found" in error_text

    def _chat_once(self, prompt: str, model: str) -> str:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid search: RRF fusion of vector similarity + BM25 keyword."""
        vector_results = self._vector_search(query, top_k=top_k * 2)
        bm25_results = self._bm25.search(query, top_k=top_k * 2)
        return self._rrf_fuse(vector_results, bm25_results, top_k=top_k)

    def _vector_search(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self.embed(query)
        results = self.store.collection.query(
            query_embeddings=[embedding], n_results=top_k
        )
        if not results["metadatas"] or not results["metadatas"][0]:
            return []
        return [
            {
                "title": m["title"],
                "path": m["path"],
                "content": d,
                "tags": m.get("tags", ""),
                "type": m.get("type", "note"),
                "language": m.get("language", ""),
                "name": m.get("name", ""),
            }
            for m, d in zip(results["metadatas"][0], results["documents"][0])
        ]

    def _rrf_fuse(
        self, vector_results: list[dict], bm25_results: list[dict], top_k: int
    ) -> list[dict]:
        """Reciprocal Rank Fusion with k=60. Deduplicates by path."""
        k = 60
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, doc in enumerate(vector_results):
            key = doc.get("path", doc.get("title", str(rank)))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs[key] = doc

        for rank, doc in enumerate(bm25_results):
            key = doc.get("path", doc.get("title", str(rank)))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in docs:
                docs[key] = doc

        ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [docs[key] for key in ranked_keys[:top_k]]

    def search_code(self, query: str, top_k: int = 5) -> list[dict]:
        """Search only code nodes (files, functions, classes)."""
        embedding = self.embed(query)
        results = self.store.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"type": {"$in": ["code_file", "function", "class"]}},
        )
        if not results["metadatas"] or not results["metadatas"][0]:
            return []
        return [
            {
                "title": m["title"],
                "path": m["path"],
                "content": d,
                "tags": m.get("tags", ""),
                "type": m.get("type", "code_file"),
                "language": m.get("language", ""),
                "name": m.get("name", ""),
            }
            for m, d in zip(results["metadatas"][0], results["documents"][0])
        ]

    def _is_meta_question(self, question: str) -> str | None:
        """Check if the question is a greeting or identity question."""
        q = question.strip().lower()
        for pattern in GREETING_PATTERNS:
            if re.match(pattern, q):
                if "who" in q or "what" in q:
                    return IDENTITY_RESPONSE
                return GREETING_RESPONSE
        return None

    def ask(self, question: str, model: str | None = None) -> str:
        answer, _ = self.ask_with_model(question, model=model)
        return answer

    def ask_with_model(self, question: str, model: str | None = None) -> tuple[str, str]:
        # Handle greetings and meta-questions without RAG
        meta = self._is_meta_question(question)
        if meta:
            return meta, RULE_BASED_MODEL

        # Vector search for semantic matches
        context_notes = self.search(question, top_k=5)

        # Extract potential entities to pull graph context
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", question)
        graph_notes = []
        for entity in entities[:3]:
            graph_notes.extend(self.store.get_graph_context(entity))

        # Combine and deduplicate
        seen_titles = {n["title"] for n in context_notes}
        for gn in graph_notes:
            if gn["title"] not in seen_titles:
                context_notes.append({
                    "title": gn["title"],
                    "path": gn["path"],
                    "content": f"[graph-linked via entity, tags: {gn.get('tags', '')}]",
                    "tags": gn.get("tags", ""),
                })
                seen_titles.add(gn["title"])

        if not context_notes:
            return (
                "I couldn't find anything relevant in your vault for that question. "
                "Try rephrasing, or check that the relevant notes are indexed.",
                RULE_BASED_MODEL,
            )

        sections = []
        for n in context_notes:
            node_type = n.get("type", "note")
            if node_type in ("code_file", "function", "class"):
                lang = n.get("language", "code").upper()
                sections.append(
                    f"# [{lang}] {n['title']}\n"
                    f"Path: {n['path']}\n\n{n['content'][:800]}"
                )
            else:
                sections.append(
                    f"# {n['title']}\n"
                    f"Path: {n['path']}\nTags: {n.get('tags', '')}\n\n{n['content'][:800]}"
                )
        context = "\n\n---\n\n".join(sections)

        prompt = f"""You are Cortex, a local AI knowledge assistant. You help the user explore and understand THEIR notes and code.

CRITICAL RULES:
- You are NOT the user. You are NOT any project mentioned in the notes. You are Cortex.
- The notes below were WRITTEN BY the user. When they ask "what am I working on?", summarize what THEIR notes say.
- Refer to the user as "you" — e.g. "You're working on..." or "Your notes mention..."
- Cite note titles when referencing specific information — e.g. "According to your note 'IAOP Architecture Vision'..."
- Results may include NOTES from the user's Obsidian vault AND CODE from their repositories.
- For code results (marked with [PYTHON], [JAVASCRIPT], etc.), reference the file path and function/class name.
- If the context doesn't contain the answer, say so honestly. Don't make things up.
- Be concise and direct. No filler. No corporate speak.
- NEVER role-play as the user or as any system/project described in the notes.

CONTEXT FROM THE USER'S VAULT AND CODE:
{context}

USER'S QUESTION: {question}

ANSWER:"""

        preferred_model = model or self.model
        resolved_model = self._resolve_chat_model(preferred_model)

        try:
            return self._chat_once(prompt, resolved_model), resolved_model
        except Exception as error:
            if not self._is_model_not_found_error(error):
                raise

            # Model cache can be stale if a model was removed after process start.
            self._resolved_model_cache.pop(preferred_model, None)
            retry_model = self._resolve_chat_model(
                preferred_model,
                force_refresh=True,
                exclude={resolved_model},
            )

            try:
                return self._chat_once(prompt, retry_model), retry_model
            except Exception as retry_error:
                if self._is_model_not_found_error(retry_error):
                    raise RuntimeError(
                        "No usable local chat model found. Pull a model with "
                        "`ollama pull qwen2.5:7b`."
                    ) from retry_error
                raise
