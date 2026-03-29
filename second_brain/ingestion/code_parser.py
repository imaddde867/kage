"""Tree-sitter based code parser for extracting structured information from source files."""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class FunctionInfo:
    """Extracted function or method information."""
    name: str
    docstring: str = ""
    params: list[str] = field(default_factory=list)
    return_type: str = ""
    line_start: int = 0
    line_end: int = 0
    calls: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Extracted class information."""
    name: str
    docstring: str = ""
    parents: list[str] = field(default_factory=list)
    methods: list[FunctionInfo] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


@dataclass
class ImportInfo:
    """Extracted import information."""
    module: str
    names: list[str] = field(default_factory=list)


@dataclass
class ParsedCodeFile:
    """All structured information extracted from a code file."""
    path: str
    filename: str
    language: str
    content: str
    checksum: str
    module_docstring: str = ""
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)


EXTENSION_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

_parsers: dict = {}
_initialized = False


def _init_parsers():
    """Lazily initialize tree-sitter parsers for each supported language."""
    global _parsers, _initialized
    if _initialized:
        return
    _initialized = True

    try:
        from tree_sitter import Language, Parser
    except ImportError:
        logger.error(
            "tree-sitter not installed. Run: "
            "pip install tree-sitter tree-sitter-python "
            "tree-sitter-javascript tree-sitter-typescript"
        )
        return

    try:
        import tree_sitter_python as tspython
        _parsers["python"] = Parser(Language(tspython.language()))
    except ImportError:
        logger.warning("tree-sitter-python not installed, Python parsing disabled")

    try:
        import tree_sitter_javascript as tsjs
        _parsers["javascript"] = Parser(Language(tsjs.language()))
    except ImportError:
        logger.warning("tree-sitter-javascript not installed, JS parsing disabled")

    try:
        import tree_sitter_typescript as tsts
        _parsers["typescript"] = Parser(Language(tsts.language_typescript()))
        _parsers["tsx"] = Parser(Language(tsts.language_tsx()))
    except ImportError:
        logger.warning("tree-sitter-typescript not installed, TS parsing disabled")


# ── Helpers ──────────────────────────────────────────────────────────

def _text(node, src: bytes) -> str:
    """Get the UTF-8 text content of a tree-sitter node."""
    return src[node.start_byte:node.end_byte].decode("utf-8")


def _strip_docstring(raw: str) -> str:
    """Strip surrounding quotes from a Python docstring."""
    for q in ('"""', "'''"):
        if raw.startswith(q) and raw.endswith(q):
            return raw[3:-3].strip()
    return raw.strip("\"'").strip()


def _extract_calls(node, src: bytes) -> list[str]:
    """Recursively extract function/method call names from an AST subtree."""
    calls: list[str] = []
    _walk_calls(node, src, calls)
    return list(set(calls))


def _walk_calls(node, src: bytes, calls: list[str]):
    # Python: call node with function child
    if node.type == "call":
        func = node.child_by_field_name("function")
        if func:
            if func.type == "identifier":
                calls.append(_text(func, src))
            elif func.type == "attribute":
                attr = func.child_by_field_name("attribute")
                if attr:
                    calls.append(_text(attr, src))
    # JS/TS: call_expression with function child
    elif node.type == "call_expression":
        func = node.child_by_field_name("function")
        if func:
            if func.type == "identifier":
                calls.append(_text(func, src))
            elif func.type == "member_expression":
                prop = func.child_by_field_name("property")
                if prop:
                    calls.append(_text(prop, src))
    for child in node.children:
        _walk_calls(child, src, calls)


# ── Python extraction ────────────────────────────────────────────────

def _py_extract_function(node, src: bytes) -> FunctionInfo:
    name_node = node.child_by_field_name("name")
    name = _text(name_node, src) if name_node else ""

    params = _py_extract_params(node.child_by_field_name("parameters"), src)

    return_type = ""
    rt_node = node.child_by_field_name("return_type")
    if rt_node:
        return_type = _text(rt_node, src)

    docstring = ""
    body = node.child_by_field_name("body")
    calls = _extract_calls(body, src) if body else []

    if body and body.child_count > 0:
        first = body.children[0]
        if first.type == "expression_statement" and first.child_count > 0:
            expr = first.children[0]
            if expr.type == "string":
                docstring = _strip_docstring(_text(expr, src))

    return FunctionInfo(
        name=name, docstring=docstring, params=params,
        return_type=return_type,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        calls=calls,
    )


def _py_extract_params(params_node, src: bytes) -> list[str]:
    if not params_node:
        return []
    names = []
    for child in params_node.children:
        if child.type == "identifier":
            p = _text(child, src)
            if p not in ("self", "cls"):
                names.append(p)
        elif child.type in (
            "typed_parameter", "default_parameter", "typed_default_parameter",
        ):
            for sub in child.children:
                if sub.type == "identifier":
                    p = _text(sub, src)
                    if p not in ("self", "cls"):
                        names.append(p)
                    break
        elif child.type == "list_splat_pattern":
            for sub in child.children:
                if sub.type == "identifier":
                    names.append("*" + _text(sub, src))
                    break
        elif child.type == "dictionary_splat_pattern":
            for sub in child.children:
                if sub.type == "identifier":
                    names.append("**" + _text(sub, src))
                    break
    return names


def _py_extract_class(node, src: bytes) -> ClassInfo:
    name_node = node.child_by_field_name("name")
    name = _text(name_node, src) if name_node else ""

    parents = []
    superclasses = node.child_by_field_name("superclasses")
    if superclasses:
        for child in superclasses.children:
            if child.type in ("identifier", "attribute"):
                parents.append(_text(child, src))

    docstring = ""
    methods = []
    body = node.child_by_field_name("body")
    if body:
        for i, child in enumerate(body.children):
            if i == 0 and child.type == "expression_statement" and child.child_count > 0:
                expr = child.children[0]
                if expr.type == "string":
                    docstring = _strip_docstring(_text(expr, src))
            if child.type == "function_definition":
                methods.append(_py_extract_function(child, src))
            elif child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type == "function_definition":
                        methods.append(_py_extract_function(sub, src))

    return ClassInfo(
        name=name, docstring=docstring, parents=parents, methods=methods,
        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
    )


def _py_extract_import(node, src: bytes) -> ImportInfo:
    named = node.named_children

    if node.type == "import_statement":
        modules = []
        for child in named:
            if child.type == "dotted_name":
                modules.append(_text(child, src))
            elif child.type == "aliased_import":
                for sub in child.named_children:
                    if sub.type == "dotted_name":
                        modules.append(_text(sub, src))
                        break
        return ImportInfo(module=modules[0] if modules else _text(node, src))

    # import_from_statement
    module = ""
    names = []
    for child in named:
        if not module and child.type in ("dotted_name", "relative_import"):
            module = _text(child, src)
        elif child.type in ("dotted_name", "identifier"):
            names.append(_text(child, src))
        elif child.type == "aliased_import":
            for sub in child.named_children:
                if sub.type in ("dotted_name", "identifier"):
                    names.append(_text(sub, src))
                    break
        elif child.type == "wildcard_import":
            names.append("*")
    return ImportInfo(module=module, names=names)


def _extract_python(tree, src: bytes):
    root = tree.root_node
    module_doc = ""
    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []
    imports: list[ImportInfo] = []
    constants: list[str] = []

    for child in root.children:
        if child.type == "expression_statement":
            if not module_doc and child.child_count > 0:
                expr = child.children[0]
                if expr.type == "string":
                    module_doc = _strip_docstring(_text(expr, src))
                    continue
            # Top-level assignment → constant
            for sub in child.children:
                if sub.type == "assignment":
                    left = sub.child_by_field_name("left")
                    if left and left.type == "identifier":
                        constants.append(_text(left, src))

        elif child.type == "function_definition":
            functions.append(_py_extract_function(child, src))

        elif child.type == "decorated_definition":
            for sub in child.children:
                if sub.type == "function_definition":
                    functions.append(_py_extract_function(sub, src))
                elif sub.type == "class_definition":
                    classes.append(_py_extract_class(sub, src))

        elif child.type == "class_definition":
            classes.append(_py_extract_class(child, src))

        elif child.type in ("import_statement", "import_from_statement"):
            imports.append(_py_extract_import(child, src))

    return module_doc, functions, classes, imports, constants


# ── JavaScript / TypeScript extraction ───────────────────────────────

def _js_extract_function(node, src: bytes) -> FunctionInfo:
    name_node = node.child_by_field_name("name")
    name = _text(name_node, src) if name_node else ""

    params = []
    params_node = node.child_by_field_name("parameters")
    if params_node:
        for child in params_node.named_children:
            if child.type == "identifier":
                params.append(_text(child, src))
            elif child.type in ("required_parameter", "optional_parameter"):
                pattern = child.child_by_field_name("pattern")
                if pattern and pattern.type == "identifier":
                    params.append(_text(pattern, src))
            elif child.type == "rest_pattern":
                for sub in child.named_children:
                    if sub.type == "identifier":
                        params.append("..." + _text(sub, src))
                        break
            elif child.type == "assignment_pattern":
                left = child.child_by_field_name("left")
                if left and left.type == "identifier":
                    params.append(_text(left, src))

    # JSDoc from previous sibling
    docstring = ""
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        t = _text(prev, src)
        if t.startswith("/**"):
            docstring = t[3:-2].strip().replace("\n * ", " ").replace(" * ", " ")

    body = node.child_by_field_name("body")
    calls = _extract_calls(body, src) if body else []

    return FunctionInfo(
        name=name, docstring=docstring, params=params,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        calls=calls,
    )


def _js_extract_class(node, src: bytes) -> ClassInfo:
    name_node = node.child_by_field_name("name")
    name = _text(name_node, src) if name_node else ""

    parents = []
    for child in node.named_children:
        if child.type in ("class_heritage", "extends_clause"):
            for sub in child.named_children:
                if sub.type == "identifier":
                    parents.append(_text(sub, src))

    docstring = ""
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        t = _text(prev, src)
        if t.startswith("/**"):
            docstring = t[3:-2].strip()

    methods = []
    body = node.child_by_field_name("body")
    if body:
        for child in body.named_children:
            if child.type == "method_definition":
                methods.append(_js_extract_function(child, src))

    return ClassInfo(
        name=name, docstring=docstring, parents=parents, methods=methods,
        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
    )


def _js_extract_import(node, src: bytes) -> ImportInfo:
    source = node.child_by_field_name("source")
    module = _text(source, src).strip("'\"") if source else ""

    names = []
    for child in node.named_children:
        if child.type == "import_clause":
            for sub in child.named_children:
                if sub.type == "identifier":
                    names.append(_text(sub, src))
                elif sub.type == "named_imports":
                    for spec in sub.named_children:
                        if spec.type == "import_specifier":
                            n = spec.child_by_field_name("name")
                            if n:
                                names.append(_text(n, src))
                elif sub.type == "namespace_import":
                    for ns in sub.named_children:
                        if ns.type == "identifier":
                            names.append("* as " + _text(ns, src))
    return ImportInfo(module=module, names=names)


def _extract_js_ts(tree, src: bytes):
    root = tree.root_node
    module_doc = ""
    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []
    imports: list[ImportInfo] = []
    constants: list[str] = []

    def _process_declaration(node, context_node=None):
        """Process a function/class/variable declaration."""
        ctx = context_node or node

        if node.type == "function_declaration":
            functions.append(_js_extract_function(node, src))
        elif node.type == "class_declaration":
            classes.append(_js_extract_class(node, src))
        elif node.type == "lexical_declaration":
            for decl in node.named_children:
                if decl.type != "variable_declarator":
                    continue
                name_node = decl.child_by_field_name("name")
                value_node = decl.child_by_field_name("value")
                if not (name_node and value_node):
                    continue
                name = _text(name_node, src)
                if value_node.type in ("arrow_function", "function"):
                    fi = _js_extract_function(value_node, src)
                    fi.name = name
                    fi.line_start = ctx.start_point[0] + 1
                    fi.line_end = ctx.end_point[0] + 1
                    prev = ctx.prev_named_sibling
                    if prev and prev.type == "comment":
                        t = _text(prev, src)
                        if t.startswith("/**"):
                            fi.docstring = t[3:-2].strip()
                    functions.append(fi)
                else:
                    constants.append(name)

    for child in root.named_children:
        if child.type == "comment" and not module_doc:
            t = _text(child, src)
            if t.startswith("/**"):
                module_doc = t[3:-2].strip()
            elif t.startswith("/*"):
                module_doc = t[2:-2].strip()
            elif t.startswith("//"):
                module_doc = t[2:].strip()

        elif child.type == "import_statement":
            imports.append(_js_extract_import(child, src))

        elif child.type == "export_statement":
            for sub in child.named_children:
                _process_declaration(sub, child)

        else:
            _process_declaration(child)

    return module_doc, functions, classes, imports, constants


# ── Public API ───────────────────────────────────────────────────────

def parse_code_file(path: Path) -> ParsedCodeFile | None:
    """Parse a code file using tree-sitter and extract structured information.

    Returns None if the file can't be parsed (unsupported extension, parse error, etc.).
    """
    ext = path.suffix.lower()
    language = EXTENSION_LANGUAGE.get(ext)
    if not language:
        return None

    _init_parsers()

    parser = _parsers.get(language)
    if not parser:
        return None

    try:
        source = path.read_bytes()
        text = source.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return None

    checksum = hashlib.md5(source).hexdigest()

    try:
        tree = parser.parse(source)
    except Exception as e:
        logger.warning(f"tree-sitter parse error for {path}: {e}")
        return None

    if language == "python":
        mod_doc, funcs, classes, imports, consts = _extract_python(tree, source)
    else:
        mod_doc, funcs, classes, imports, consts = _extract_js_ts(tree, source)

    return ParsedCodeFile(
        path=str(path),
        filename=path.name,
        language=language,
        content=text,
        checksum=checksum,
        module_docstring=mod_doc,
        functions=funcs,
        classes=classes,
        imports=imports,
        constants=consts,
    )
