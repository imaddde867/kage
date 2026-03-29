import hashlib

import kuzu
import chromadb
from pathlib import Path
from ingestion.parser import ParsedNote
from ingestion.code_parser import ParsedCodeFile


class BrainStore:
    def __init__(self, data_dir: str = "./data"):
        Path(data_dir).mkdir(exist_ok=True)
        self.db = kuzu.Database(f"{data_dir}/graph.kuzu")
        self.conn = kuzu.Connection(self.db)
        self.chroma = chromadb.PersistentClient(path=f"{data_dir}/chroma")
        self.collection = self.chroma.get_or_create_collection("notes")
        self._init_schema()

    def _init_schema(self):
        tables = [
            """CREATE NODE TABLE IF NOT EXISTS Note (
                id STRING, title STRING, path STRING,
                checksum STRING, tags STRING,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Tag (
                name STRING, PRIMARY KEY (name)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Entity (
                name STRING, PRIMARY KEY (name)
            )""",
            """CREATE REL TABLE IF NOT EXISTS TAGGED (FROM Note TO Tag)""",
            """CREATE REL TABLE IF NOT EXISTS LINKS_TO (FROM Note TO Entity)""",
            """CREATE REL TABLE IF NOT EXISTS RELATED_TO (FROM Note TO Note, weight DOUBLE)""",
            # ── Code tables ──────────────────────────────────────
            """CREATE NODE TABLE IF NOT EXISTS Repository (
                id STRING, name STRING, path STRING,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS CodeFile (
                id STRING, path STRING, filename STRING,
                language STRING, repo_id STRING,
                checksum STRING,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Function (
                id STRING, name STRING, file_id STRING,
                docstring STRING, params STRING,
                line_start INT64, line_end INT64,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Class (
                id STRING, name STRING, file_id STRING,
                docstring STRING, parents STRING,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Module (
                name STRING, PRIMARY KEY (name)
            )""",
            # ── Code relationships ───────────────────────────────
            """CREATE REL TABLE IF NOT EXISTS CONTAINS (FROM CodeFile TO Function)""",
            """CREATE REL TABLE IF NOT EXISTS CONTAINS_CLASS (FROM CodeFile TO Class)""",
            """CREATE REL TABLE IF NOT EXISTS HAS_METHOD (FROM Class TO Function)""",
            """CREATE REL TABLE IF NOT EXISTS IMPORTS (FROM CodeFile TO Module)""",
            """CREATE REL TABLE IF NOT EXISTS IN_REPO (FROM CodeFile TO Repository)""",
            """CREATE REL TABLE IF NOT EXISTS CALLS (FROM Function TO Function)""",
        ]
        for ddl in tables:
            self.conn.execute(ddl)

    def upsert_note(self, note: ParsedNote, embedding: list[float]):
        note_id = note.checksum

        # Upsert the note node
        self.conn.execute("""
            MERGE (n:Note {id: $id})
            SET n.title = $title, n.path = $path,
                n.checksum = $checksum, n.tags = $tags
        """, {
            "id": note_id, "title": note.title, "path": note.path,
            "checksum": note.checksum, "tags": ",".join(note.tags),
        })

        # Clear old relationships for this note so re-indexing is clean
        self.conn.execute(
            "MATCH (n:Note {id: $id})-[r:TAGGED]->() DELETE r",
            {"id": note_id},
        )
        self.conn.execute(
            "MATCH (n:Note {id: $id})-[r:LINKS_TO]->() DELETE r",
            {"id": note_id},
        )

        # Create tag nodes + relationships
        for tag in note.tags:
            self.conn.execute(
                "MERGE (t:Tag {name: $name})", {"name": tag}
            )
            self.conn.execute("""
                MATCH (n:Note {id: $nid}), (t:Tag {name: $tname})
                CREATE (n)-[:TAGGED]->(t)
            """, {"nid": note_id, "tname": tag})

        # Create entity nodes + relationships
        for link in note.links:
            self.conn.execute(
                "MERGE (e:Entity {name: $name})", {"name": link}
            )
            self.conn.execute("""
                MATCH (n:Note {id: $nid}), (e:Entity {name: $ename})
                CREATE (n)-[:LINKS_TO]->(e)
            """, {"nid": note_id, "ename": link})

        # Upsert into ChromaDB for vector search
        self.collection.upsert(
            ids=[note_id],
            embeddings=[embedding],
            documents=[note.content[:2000]],
            metadatas=[{
                "title": note.title,
                "path": note.path,
                "tags": ",".join(note.tags),
            }],
        )

    def get_graph_context(self, entity_name: str) -> list[dict]:
        """Find notes connected to a given entity via the graph."""
        try:
            result = self.conn.execute("""
                MATCH (n:Note)-[:LINKS_TO]->(e:Entity {name: $name})
                RETURN n.title, n.path, n.tags
            """, {"name": entity_name})
            return [
                {"title": row[0], "path": row[1], "tags": row[2]}
                for row in result.get_as_df().itertuples(index=False)
            ]
        except Exception:
            return []

    def get_related_by_tag(self, tag: str) -> list[dict]:
        """Find all notes with a given tag via the graph."""
        try:
            result = self.conn.execute("""
                MATCH (n:Note)-[:TAGGED]->(t:Tag {name: $name})
                RETURN n.title, n.path, n.id
            """, {"name": tag})
            return [
                {"title": row[0], "path": row[1], "id": row[2]}
                for row in result.get_as_df().itertuples(index=False)
            ]
        except Exception:
            return []

    # ── Code ingestion methods ───────────────────────────────────────

    def get_code_checksum(self, file_path: str) -> str | None:
        """Get the stored checksum for a code file, or None if not indexed."""
        try:
            result = self.conn.execute(
                "MATCH (f:CodeFile {path: $path}) RETURN f.checksum",
                {"path": file_path},
            )
            if result.has_next():
                return result.get_next()[0]
            return None
        except Exception:
            return None

    def _clear_code_file_data(self, file_id: str):
        """Remove old nodes and relationships for a code file before re-indexing."""
        # Delete methods of classes contained by this file
        self.conn.execute(
            "MATCH (f:CodeFile {id: $id})-[:CONTAINS_CLASS]->(c:Class)"
            "-[:HAS_METHOD]->(fn:Function) DETACH DELETE fn",
            {"id": file_id},
        )
        # Delete classes
        self.conn.execute(
            "MATCH (f:CodeFile {id: $id})-[:CONTAINS_CLASS]->(c:Class) DETACH DELETE c",
            {"id": file_id},
        )
        # Delete module-level functions
        self.conn.execute(
            "MATCH (f:CodeFile {id: $id})-[:CONTAINS]->(fn:Function) DETACH DELETE fn",
            {"id": file_id},
        )
        # Delete relationships from the file
        for rel in ("IMPORTS", "IN_REPO"):
            self.conn.execute(
                f"MATCH (f:CodeFile {{id: $id}})-[r:{rel}]->() DELETE r",
                {"id": file_id},
            )

    def upsert_code_file(
        self,
        parsed: ParsedCodeFile,
        embeddings: dict,
        repo_id: str,
        repo_name: str,
        repo_path: str,
    ):
        """Upsert a parsed code file into the graph and vector store."""
        file_id = hashlib.md5(parsed.path.encode()).hexdigest()

        # Upsert Repository
        self.conn.execute(
            "MERGE (r:Repository {id: $id}) SET r.name = $name, r.path = $path",
            {"id": repo_id, "name": repo_name, "path": repo_path},
        )

        # Upsert CodeFile
        self.conn.execute("""
            MERGE (f:CodeFile {id: $id})
            SET f.path = $path, f.filename = $filename,
                f.language = $language, f.repo_id = $repo_id,
                f.checksum = $checksum
        """, {
            "id": file_id, "path": parsed.path, "filename": parsed.filename,
            "language": parsed.language, "repo_id": repo_id,
            "checksum": parsed.checksum,
        })

        # Clear old data for this file
        self._clear_code_file_data(file_id)

        # IN_REPO
        self.conn.execute("""
            MATCH (f:CodeFile {id: $fid}), (r:Repository {id: $rid})
            CREATE (f)-[:IN_REPO]->(r)
        """, {"fid": file_id, "rid": repo_id})

        # ── Functions ────────────────────────────────────────────
        all_func_ids: dict[str, str] = {}

        for func in parsed.functions:
            func_id = hashlib.md5(
                f"{parsed.path}::{func.name}".encode()
            ).hexdigest()
            all_func_ids[func.name] = func_id

            self.conn.execute("""
                MERGE (fn:Function {id: $id})
                SET fn.name = $name, fn.file_id = $file_id,
                    fn.docstring = $docstring, fn.params = $params,
                    fn.line_start = $ls, fn.line_end = $le
            """, {
                "id": func_id, "name": func.name, "file_id": file_id,
                "docstring": func.docstring[:500],
                "params": ",".join(func.params),
                "ls": func.line_start, "le": func.line_end,
            })

            self.conn.execute("""
                MATCH (f:CodeFile {id: $fid}), (fn:Function {id: $fnid})
                CREATE (f)-[:CONTAINS]->(fn)
            """, {"fid": file_id, "fnid": func_id})

            if func.name in embeddings.get("functions", {}):
                self.collection.upsert(
                    ids=[func_id],
                    embeddings=[embeddings["functions"][func.name]],
                    documents=[f"{func.name}: {func.docstring[:500]}"],
                    metadatas=[{
                        "title": f"function: {func.name}",
                        "path": parsed.path,
                        "type": "function",
                        "language": parsed.language,
                        "repo": repo_name,
                        "name": func.name,
                        "tags": "",
                    }],
                )

        # ── Classes ──────────────────────────────────────────────
        for cls in parsed.classes:
            class_id = hashlib.md5(
                f"{parsed.path}::{cls.name}".encode()
            ).hexdigest()

            self.conn.execute("""
                MERGE (c:Class {id: $id})
                SET c.name = $name, c.file_id = $file_id,
                    c.docstring = $docstring, c.parents = $parents
            """, {
                "id": class_id, "name": cls.name, "file_id": file_id,
                "docstring": cls.docstring[:500],
                "parents": ",".join(cls.parents),
            })

            self.conn.execute("""
                MATCH (f:CodeFile {id: $fid}), (c:Class {id: $cid})
                CREATE (f)-[:CONTAINS_CLASS]->(c)
            """, {"fid": file_id, "cid": class_id})

            # Methods
            for method in cls.methods:
                method_id = hashlib.md5(
                    f"{parsed.path}::{cls.name}::{method.name}".encode()
                ).hexdigest()
                all_func_ids[method.name] = method_id

                self.conn.execute("""
                    MERGE (fn:Function {id: $id})
                    SET fn.name = $name, fn.file_id = $file_id,
                        fn.docstring = $docstring, fn.params = $params,
                        fn.line_start = $ls, fn.line_end = $le
                """, {
                    "id": method_id, "name": method.name, "file_id": file_id,
                    "docstring": method.docstring[:500],
                    "params": ",".join(method.params),
                    "ls": method.line_start, "le": method.line_end,
                })

                self.conn.execute("""
                    MATCH (c:Class {id: $cid}), (fn:Function {id: $fnid})
                    CREATE (c)-[:HAS_METHOD]->(fn)
                """, {"cid": class_id, "fnid": method_id})

                method_key = f"{cls.name}::{method.name}"
                if method_key in embeddings.get("methods", {}):
                    self.collection.upsert(
                        ids=[method_id],
                        embeddings=[embeddings["methods"][method_key]],
                        documents=[
                            f"{cls.name}.{method.name}: {method.docstring[:500]}"
                        ],
                        metadatas=[{
                            "title": f"method: {cls.name}.{method.name}",
                            "path": parsed.path,
                            "type": "function",
                            "language": parsed.language,
                            "repo": repo_name,
                            "name": f"{cls.name}.{method.name}",
                            "tags": "",
                        }],
                    )

            # Class embedding
            if cls.name in embeddings.get("classes", {}):
                self.collection.upsert(
                    ids=[class_id],
                    embeddings=[embeddings["classes"][cls.name]],
                    documents=[f"class {cls.name}: {cls.docstring[:500]}"],
                    metadatas=[{
                        "title": f"class: {cls.name}",
                        "path": parsed.path,
                        "type": "class",
                        "language": parsed.language,
                        "repo": repo_name,
                        "name": cls.name,
                        "tags": "",
                    }],
                )

        # ── Imports → Module nodes ───────────────────────────────
        for imp in parsed.imports:
            if not imp.module:
                continue
            self.conn.execute(
                "MERGE (m:Module {name: $name})", {"name": imp.module}
            )
            self.conn.execute("""
                MATCH (f:CodeFile {id: $fid}), (m:Module {name: $mname})
                CREATE (f)-[:IMPORTS]->(m)
            """, {"fid": file_id, "mname": imp.module})

        # ── CALLS relationships (same-file resolution) ───────────
        all_functions = list(parsed.functions)
        for cls in parsed.classes:
            all_functions.extend(cls.methods)

        for func in all_functions:
            caller_id = all_func_ids.get(func.name)
            if not caller_id:
                continue
            for call_name in func.calls:
                target_id = all_func_ids.get(call_name)
                if target_id and target_id != caller_id:
                    try:
                        self.conn.execute("""
                            MATCH (a:Function {id: $aid}), (b:Function {id: $bid})
                            CREATE (a)-[:CALLS]->(b)
                        """, {"aid": caller_id, "bid": target_id})
                    except Exception:
                        pass

        # ── File-level embedding ─────────────────────────────────
        self.collection.upsert(
            ids=[file_id],
            embeddings=[embeddings["file"]],
            documents=[parsed.content[:2000]],
            metadatas=[{
                "title": f"file: {parsed.filename}",
                "path": parsed.path,
                "type": "code_file",
                "language": parsed.language,
                "repo": repo_name,
                "name": parsed.filename,
                "tags": "",
            }],
        )

    def get_code_stats(self) -> dict:
        """Get counts of code-related graph nodes."""
        stats = {}
        for label, query in [
            ("repositories", "MATCH (r:Repository) RETURN count(r)"),
            ("code_files", "MATCH (f:CodeFile) RETURN count(f)"),
            ("functions", "MATCH (fn:Function) RETURN count(fn)"),
            ("classes", "MATCH (c:Class) RETURN count(c)"),
            ("modules", "MATCH (m:Module) RETURN count(m)"),
        ]:
            try:
                result = self.conn.execute(query)
                stats[label] = result.get_next()[0]
            except Exception:
                stats[label] = 0
        return stats
