"""The agent skill in plugins/gotranx: manifests, portability, and that everything the skill names
exists (gotranx objects and their arguments, repository links, CLI commands, options and option
values, scheme names, and the API of the generated code)."""

import ast
import enum
import importlib
import inspect
import json
import re
import tomllib
import typing
from pathlib import Path

import attrs
import pytest
import typer.main

ROOT = Path(__file__).resolve().parents[1]
REPO = "gotranx"
OWNER = "finsberg"
SKILL_NAME = "gotranx-odes"
IMPORT_NAME = "gotranx"
CLI_PROG = "gotranx"
CLI_ENTRY = "gotranx.cli:app"
FORBIDDEN_TOOL_NAMES = ("AskUserQuestion", "TodoWrite", "Skill tool", "Agent tool", "subagent_type")
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"

# Inline code may wrap onto the next line, but not across a blank line (a new paragraph).
INLINE_CODE = re.compile(r"`((?:[^`\n]|\n(?![ \t]*\n))+)`")
FENCED_CODE = re.compile(r"```([^\n]*)\n(.*?)```", flags=re.S)


def _marketplace() -> dict:
    return json.loads(MARKETPLACE.read_text(encoding="utf-8"))


def _plugin_dir() -> Path:
    (entry,) = _marketplace()["plugins"]
    return (ROOT / entry["source"]).resolve()


def _skill_dir() -> Path:
    return _plugin_dir() / "skills" / SKILL_NAME


def _skill_files() -> list[Path]:
    return sorted(_skill_dir().glob("*.md"))


def _skill_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in _skill_files())


def _fenced(text: str) -> list[tuple[str, str]]:
    """(language, body) of every fenced code block."""
    return [(lang.strip(), body) for lang, body in FENCED_CODE.findall(text)]


def _inline(text: str) -> list[str]:
    """Inline code outside fenced blocks, with a wrapped line joined by a space."""
    without_blocks = FENCED_CODE.sub("", text)
    return [re.sub(r"\s*\n\s*", " ", span) for span in INLINE_CODE.findall(without_blocks)]


def _code_spans(text: str) -> list[str]:
    """Fenced code blocks and inline code: where the skill names API, commands and paths."""
    return [body for _, body in _fenced(text)] + _inline(text)


def _frontmatter(text: str) -> dict[str, str]:
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    block = text.split("---\n", 2)[1]
    out = {}
    for line in block.splitlines():
        if line.strip():
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


def test_marketplace_manifest():
    data = _marketplace()
    assert data["name"] == REPO
    assert data["owner"]["name"]
    (entry,) = data["plugins"]
    assert entry["name"] == REPO
    assert entry["source"] == f"./plugins/{REPO}"
    assert (_plugin_dir() / ".claude-plugin" / "plugin.json").is_file()


def test_plugin_manifest_version_matches_package():
    plugin = json.loads(
        (_plugin_dir() / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    assert plugin["name"] == REPO
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert plugin["version"] == project["version"]


def test_skill_frontmatter_is_portable_and_small():
    text = (_skill_dir() / "SKILL.md").read_text(encoding="utf-8")
    meta = _frontmatter(text)
    assert set(meta) <= {"name", "description", "license", "metadata"}, set(meta)
    assert meta["name"] == SKILL_NAME
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) and len(meta["name"]) <= 64
    assert 0 < len(meta["description"]) <= 1024
    assert len(text.splitlines()) < 500
    assert (_skill_dir() / "gotchas.md").is_file()


def test_skill_portable_wording():
    text = _skill_text()
    hits = [name for name in FORBIDDEN_TOOL_NAMES if name in text]
    assert not hits, f"agent-specific tool names in the skill: {hits}"


def test_skill_relative_links_resolve():
    for path in _skill_files():
        for target in re.findall(r"\]\(([^)\s#]+)(?:#[^)]*)?\)", path.read_text(encoding="utf-8")):
            if re.match(r"[a-z]+:", target):
                continue
            assert (path.parent / target).exists(), f"{path.name}: broken link {target}"


def _anchor(heading: str) -> str:
    """GitHub's anchor for a markdown heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_skill_anchor_links_resolve():
    for path in _skill_files():
        for target, anchor in re.findall(
            r"\]\(([^)\s#]*)#([^)\s]+)\)", path.read_text(encoding="utf-8")
        ):
            if re.match(r"[a-z]+:", target):
                continue
            linked = path.parent / target if target else path
            headings = re.findall(
                r"^#+ (.+)$", FENCED_CODE.sub("", linked.read_text(encoding="utf-8")), flags=re.M
            )
            assert anchor in {_anchor(h) for h in headings}, (
                f"{path.name}: no heading for #{anchor}"
            )


def test_skill_repo_links_exist():
    """Every link to a file in this repository on GitHub points to a file that exists."""
    pattern = rf"github\.com/{OWNER}/{REPO}/(?:blob|tree)/main/([^)\s>#`]+)"
    paths = set(re.findall(pattern, _skill_text()))
    missing = sorted(p for p in paths if not (ROOT / p.rstrip("/")).exists())
    assert not missing, f"links to files that do not exist: {missing}"


def _resolve(dotted: str) -> None:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return
    raise ImportError(dotted)


def test_skill_package_names_resolve():
    pattern = rf"(?<![\w/.-])({IMPORT_NAME}(?:\.[A-Za-z_]\w*)+)"
    names = {n for span in _code_spans(_skill_text()) for n in re.findall(pattern, span)}
    assert names, f"the skill should name {IMPORT_NAME} objects"
    bad = []
    for name in sorted(names):
        try:
            _resolve(name)
        except (ImportError, AttributeError):
            bad.append(name)
    assert not bad, f"unknown {IMPORT_NAME} names: {bad}"


# --- Python snippets: imports, attributes, methods and keyword arguments ---

_MISSING = object()
_FIELD = object()


def _member(cls: type, name: str):
    """`name` on `cls` or on one of its subclasses: return annotations may name a base class
    (`ODE.get_component` returns a `BaseComponent`; the object is a `Component`)."""
    stack, seen = [cls], set()
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        try:
            return inspect.getattr_static(c, name)
        except AttributeError:
            pass
        if attrs.has(c) and name in attrs.fields_dict(c):
            return _FIELD
        stack.extend(c.__subclasses__())
    return _MISSING


def _unwrap(obj):
    if isinstance(obj, property):
        return obj.fget
    if isinstance(obj, (staticmethod, classmethod)):
        return obj.__func__
    return obj


def _returns(obj):
    """The package class that a function, method or property is annotated to return."""
    try:
        hint = typing.get_type_hints(_unwrap(obj)).get("return")
    except Exception:
        return None
    if inspect.isclass(hint) and hint.__module__.split(".")[0] == IMPORT_NAME:
        return ("type", hint)
    return None


class _ApiCheck(ast.NodeVisitor):
    """Follows package objects through a snippet: names bound by imports and assignments, and the
    annotated return types of calls, so that `ode = gotranx.load_ode(...)` makes `ode.split(...)`
    a call of `ODE.split`. Every attribute and keyword argument of such an object must exist."""

    def __init__(self, where: str):
        self.where = where
        # inline code such as `gotranx.load_ode(path, ...)` uses the package without importing it
        self.env: dict[str, tuple[str, object]] = {
            IMPORT_NAME: ("obj", importlib.import_module(IMPORT_NAME))
        }
        self.bad: list[str] = []

    def error(self, node: ast.AST, message: str) -> None:
        self.bad.append(f"{self.where}: `{ast.unparse(node)}`: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] != IMPORT_NAME:
                continue
            try:
                module = importlib.import_module(alias.name)
            except ImportError:
                self.error(node, f"no module {alias.name}")
                continue
            if alias.asname:
                self.env[alias.asname] = ("obj", module)
            else:
                self.env[IMPORT_NAME] = ("obj", importlib.import_module(IMPORT_NAME))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None or node.module.split(".")[0] != IMPORT_NAME:
            return
        try:
            module = importlib.import_module(node.module)
        except ImportError:
            self.error(node, f"no module {node.module}")
            return
        for alias in node.names:
            value = self.attribute(("obj", module), alias.name, node)
            if value is not None:
                self.env[alias.asname or alias.name] = value

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self.evaluate(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if value is None:
                    self.env.pop(target.id, None)
                else:
                    self.env[target.id] = value
            else:
                self.evaluate(target)

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.evaluate(child)
            else:
                self.visit(child)

    def attribute(self, base, name: str, node: ast.AST):
        kind, value = base
        if kind == "obj":
            if hasattr(value, name):
                return ("obj", getattr(value, name))
            if inspect.ismodule(value):
                try:
                    return ("obj", importlib.import_module(f"{value.__name__}.{name}"))
                except ImportError:
                    pass
            self.error(node, f"{getattr(value, '__name__', value)!s} has no attribute {name}")
            return None
        if kind == "type":
            member = _member(value, name)
            if member is _MISSING:
                self.error(node, f"{value.__name__} has no attribute {name}")
                return None
            if member is _FIELD:
                return None
            if isinstance(member, property):
                return _returns(member)
            return ("member", member)
        return None

    def check_keywords(self, func, node: ast.Call) -> None:
        try:
            parameters = inspect.signature(_unwrap(func)).parameters
        except (TypeError, ValueError):
            return
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg not in parameters:
                self.error(node, f"no argument {keyword.arg}")

    def evaluate(self, node: ast.expr):
        if isinstance(node, ast.Name):
            return self.env.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.evaluate(node.value)
            return None if base is None else self.attribute(base, node.attr, node)
        if isinstance(node, ast.Call):
            func = self.evaluate(node.func)
            for arg in node.args:
                self.evaluate(arg)
            for keyword in node.keywords:
                self.evaluate(keyword.value)
            if func is None:
                return None
            kind, value = func
            if kind in ("obj", "member") and callable(_unwrap(value)):
                self.check_keywords(value, node)
                if inspect.isclass(value):
                    return None
                return _returns(value)
            return None
        if isinstance(node, ast.BinOp):
            left = self.evaluate(node.left)
            self.evaluate(node.right)
            if left and left[0] == "type" and isinstance(node.op, ast.Sub):
                method = _member(left[1], "__sub__")
                return None if method in (_MISSING, _FIELD) else _returns(method)
            return None
        self.generic_visit(node)
        return None


def _python_snippets() -> list[tuple[str, str, bool]]:
    """(where, source, must_parse): python blocks, `python -c "..."` commands, and inline code
    that names the package."""
    out = []
    for path in _skill_files():
        text = path.read_text(encoding="utf-8")
        for lang, body in _fenced(text):
            if lang == "python":
                out.append((f"{path.name} python block", body, True))
        for span in _code_spans(text):
            for command in re.findall(r"python3? -c \"([^\"]+)\"", span):
                out.append((f"{path.name} python -c", command, True))
        for span in _inline(text):
            if IMPORT_NAME in span and "python -c" not in span:
                out.append((f"{path.name} inline", span, False))
    return out


def test_skill_python_snippets_use_existing_api():
    snippets = _python_snippets()
    assert any(must for _, _, must in snippets), "the skill should show python code"
    bad = []
    for where, source, must_parse in snippets:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            if must_parse:
                bad.append(f"{where}: does not parse: {e}")
            continue
        check = _ApiCheck(where)
        check.visit(tree)
        bad.extend(check.bad)
    assert not bad, "unknown API in python snippets:\n" + "\n".join(bad)


def test_skill_scheme_names_exist():
    from gotranx.schemes import Scheme

    known = {s.value for s in Scheme}
    named = {
        span
        for span in _inline(_skill_text())
        if re.fullmatch(r"[a-z0-9_]+", span) and ("rush_larsen" in span or "euler" in span)
    }
    assert named, "the skill should name schemes"
    assert named <= known, f"unknown schemes: {sorted(named - known)}"


# --- CLI: commands, options and option values ---


def _cli_root():
    """The click command tree. Duck-typed: typer may bring its own copy of click."""
    module, _, attr = CLI_ENTRY.partition(":")
    obj = getattr(importlib.import_module(module), attr)
    if hasattr(obj, "params"):
        return obj
    return typer.main.get_command(obj)


def _invocations(text: str) -> list[list[str]]:
    """Token lists after the program name, where the program is the command being run: at the
    start of a line, after a shell separator, or after `python -m`. A line that starts with a
    subcommand (`ode2c --to .h`) counts as an invocation too."""
    start = r"(?:^|&&|\|\||[;|(]|\$\(|python3? -m)\s*"
    commands = set(getattr(_cli_root(), "commands", {}))
    out = []
    for span in _code_spans(text):
        for line in span.splitlines():
            line = line.split(" #", 1)[0].strip()
            found = False
            for m in re.finditer(rf"{start}{re.escape(CLI_PROG)}((?:[ \t]+[^\s;&|)]+)*)", line):
                out.append(m.group(1).split())
                found = True
            if not found and line.split()[:1] and line.split()[0] in commands:
                out.append(line.split())
    return out


def _bare_options(text: str) -> list[list[str]]:
    """Inline code that is only options, such as `--shape single` or `--cse/--no-cse`."""
    return [span.split() for span in _inline(text) if span.startswith("-")]


def _option_map(cmd) -> dict:
    out = {o: p for p in cmd.params for o in (*p.opts, *p.secondary_opts)}
    out.setdefault("--help", None)
    return out


def _takes_value(param) -> bool:
    return (
        param is not None
        and getattr(param, "param_type_name", "") == "option"
        and not getattr(param, "is_flag", False)
        and not getattr(param, "count", False)
    )


def _enum_in(annotation) -> type[enum.Enum] | None:
    """The Enum in an annotation such as `Scheme`, `Optional[List[Scheme]]` or `Annotated[...]`."""
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return annotation
    for arg in typing.get_args(annotation):
        found = _enum_in(arg)
        if found is not None:
            return found
    return None


def _choices(param) -> list[str] | None:
    """Allowed values of an option, or None for any value. click keeps them on `param.type`; the
    click that typer >= 0.29 bundles keeps the annotation on `param.type_descriptor` instead."""
    choices = getattr(getattr(param, "type", None), "choices", None)
    if choices is None:
        annotation = getattr(getattr(param, "type_descriptor", None), "annotation", None)
        found = _enum_in(annotation)
        choices = list(found) if found is not None else None
    return None if choices is None else [str(getattr(c, "value", c)) for c in choices]


def _check_options(tokens: list[str], option_maps: list[dict], label: str) -> list[str]:
    """Every option is in one of `option_maps`, and a value given to it is one of its choices."""
    bad = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if not tok.startswith("-") or tok in {"-", "--"}:
            continue
        name, eq, inline_value = tok.partition("=")
        for opt in name.split("/"):
            params = [m[opt] for m in option_maps if opt in m]
            if not params:
                bad.append(f"{label}: {opt}")
                continue
            if not any(_takes_value(p) for p in params):
                continue
            value = inline_value if eq else (tokens[i] if i < len(tokens) else None)
            if not eq and value is not None:
                i += 1
            if value is None or value.startswith(("<", "-")) or value in {"...", "…"}:
                continue
            allowed = [_choices(p) for p in params if _takes_value(p)]
            if any(a is None for a in allowed):
                continue
            if all(value not in a for a in allowed if a is not None):
                valid = sorted({c for a in allowed if a is not None for c in a})
                bad.append(f"{label}: {opt} {value} (choices: {valid})")
    return bad


def test_skill_cli_invocations_exist():
    root = _cli_root()
    calls = _invocations(_skill_text())
    assert calls, f"the skill should show {CLI_PROG} commands"
    bad = []
    for tokens in calls:
        cmd = root
        rest = []
        for j, tok in enumerate(tokens):
            if tok.startswith("<") or tok in {"...", "…"}:
                continue
            if not tok.startswith("-") and hasattr(cmd, "commands"):
                if tok not in cmd.commands:
                    bad.append(f"{' '.join(tokens)}: no subcommand {tok}")
                    break
                cmd = cmd.commands[tok]
                continue
            rest = tokens[j:]
            break
        bad.extend(_check_options(rest, [_option_map(cmd)], " ".join(tokens)))
    assert not bad, f"unknown CLI commands/options: {bad}"


def test_skill_bare_options_exist():
    """An option shown on its own exists in some command, with a valid value if it has one."""
    root = _cli_root()
    maps = [_option_map(root)] + [_option_map(c) for c in root.commands.values()]
    spans = _bare_options(_skill_text())
    assert spans, "the skill should name options"
    bad = [b for tokens in spans for b in _check_options(tokens, maps, " ".join(tokens))]
    assert not bad, f"unknown options: {bad}"


# --- the API of the generated code ---

TINY_ODE = """
parameters("a", k=1.0)
states("a", x=1.0)
expressions("a")
y_out = 2*x
dx_dt = -k*x

states("b", z=0.5)
expressions("b")
w = 3*z
dz_dt = -z + y_out
"""
SCHEME_PLACEHOLDERS = {"<scheme>", "scheme"}
SIGNATURE = re.compile(r"(<scheme>|[A-Za-z_]\w*)\(([^()]*)\)")


def _sections(path: Path, start: str, stop: str | None) -> str:
    text = path.read_text(encoding="utf-8")
    begin = text.index(start)
    end = text.index(stop, begin) if stop else len(text)
    return text[begin:end]


def _python_api_text() -> str:
    """Skill text about generated Python: everything except the other backends in targets.md."""
    targets = _skill_dir() / "targets.md"
    rest = "\n".join(p.read_text(encoding="utf-8") for p in _skill_files() if p != targets)
    return rest + _sections(targets, "## Python", "## C ")


def _c_api_text() -> str:
    return _sections(_skill_dir() / "targets.md", "## C ", "## C++")


@pytest.fixture(scope="module")
def generated():
    """Generated Python modules (a whole model, a split half without missing variables, and one
    with) and generated C for the whole model."""
    import gotranx
    from gotranx.codegen.c import Format as CFormat
    from gotranx.codegen.python import Format
    from gotranx.load import ode_from_string
    from gotranx.schemes import Scheme

    ode = ode_from_string(TINY_ODE, name="tiny")
    split = ode.split("half", inputs=["y_out"], outputs=["w"])
    component = split.get_component("half")
    half = component.to_ode()
    rest = split - component
    assert half.missing_variables and not rest.missing_variables

    modules = []
    for model, missing in [
        (ode, None),
        (rest, half.missing_variables),
        (half, rest.missing_variables),
    ]:
        namespace: dict = {}
        code = gotranx.cli.gotran2py.get_code(
            model, scheme=[Scheme.rush_larsen], format=Format.none, missing_values=missing
        )
        exec(code, namespace)
        modules.append(namespace)
    c_code = gotranx.cli.gotran2c.get_code(ode, scheme=[Scheme.rush_larsen], format=CFormat.none)
    return {"python": modules, "c": c_code, "scheme": Scheme.rush_larsen.value}


def _documented_signatures(text: str, names: set[str]) -> list[tuple[str, list[str]]]:
    out = []
    for span in _inline(text):
        m = SIGNATURE.fullmatch(span)
        if m and (m.group(1) in names or m.group(1) in SCHEME_PLACEHOLDERS):
            args = [a.strip() for a in m.group(2).split(",") if a.strip()]
            if not any("..." in a or "…" in a or "<" in a for a in args):
                out.append((m.group(1), args))
    return out


def _matches_python(func, args: list[str]) -> bool:
    params = list(inspect.signature(func).parameters.values())
    var_keyword = any(p.kind is p.VAR_KEYWORD for p in params)
    positional = [p.name for p in params if p.kind is not p.VAR_KEYWORD]
    names = []
    for arg in args:
        if arg.startswith("**") or "=" in arg:
            if not var_keyword:
                return False
        else:
            names.append(arg)
    return names == positional or (not names and var_keyword)


def test_skill_generated_python_api(generated):
    modules = generated["python"]
    functions = {k for m in modules for k, v in m.items() if inspect.isfunction(v)}
    documented = _documented_signatures(_python_api_text(), functions)
    assert documented, "the skill should document the generated functions"
    bad = []
    for name, args in documented:
        real = generated["scheme"] if name in SCHEME_PLACEHOLDERS else name
        candidates = [m[real] for m in modules if real in m]
        if not any(_matches_python(f, args) for f in candidates):
            found = sorted({f"{real}{inspect.signature(f)}" for f in candidates}) or ["none"]
            bad.append(f"{name}({', '.join(args)}): generated {found}")
    # names in the table of the generated module
    table = [
        line for line in _sections(_skill_dir() / "targets.md", "## Python", "## C ").splitlines()
    ]
    for line in table:
        if line.startswith("|"):
            for span in INLINE_CODE.findall(line.split("|")[1]):
                name = span.split("(")[0]
                if name not in SCHEME_PLACEHOLDERS and not any(name in m for m in modules):
                    bad.append(f"`{span}` is not in the generated module")
    assert not bad, f"documented generated Python API differs: {bad}"


def test_skill_generated_c_api(generated):
    source = generated["c"]
    defined = {
        name: [re.findall(r"\w+", a.replace("[]", ""))[-1] for a in args.split(",") if a.strip()]
        for name, args in re.findall(
            r"^[A-Za-z][\w \t*]*?\b(\w+)\(([^)]*)\)\s*\{", source, flags=re.M
        )
    }
    text = _c_api_text()
    documented = _documented_signatures(text, set(defined))
    assert documented, "the skill should document the generated C functions"
    bad = []
    for name, args in documented:
        real = generated["scheme"] if name in SCHEME_PLACEHOLDERS else name
        names = [re.findall(r"\w+", a.replace("[]", ""))[-1] for a in args]
        if defined.get(real) != names:
            bad.append(
                f"{name}({', '.join(args)}): generated {real}({', '.join(defined.get(real, []))})"
            )
    for span in _inline(text):
        if re.fullmatch(r"[A-Z][A-Z_]+", span) and not re.search(rf"\b{span}\b", source):
            bad.append(f"`{span}` is not in the generated C")
    assert not bad, f"documented generated C API differs: {bad}"
