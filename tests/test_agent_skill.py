"""The agent skill in plugins/gotranx: manifests, portability, and that everything the skill names
exists (gotranx objects, repository links, CLI commands and options)."""

import importlib
import json
import re
import tomllib
from pathlib import Path

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


def _marketplace() -> dict:
    return json.loads(MARKETPLACE.read_text())


def _plugin_dir() -> Path:
    (entry,) = _marketplace()["plugins"]
    return (ROOT / entry["source"]).resolve()


def _skill_dir() -> Path:
    return _plugin_dir() / "skills" / SKILL_NAME


def _skill_files() -> list[Path]:
    return sorted(_skill_dir().glob("*.md"))


def _skill_text() -> str:
    return "\n".join(p.read_text() for p in _skill_files())


def _code_spans(text: str) -> list[str]:
    """Fenced code blocks and inline code: where the skill names API, commands and paths."""
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S)
    without_blocks = re.sub(r"```.*?```", "", text, flags=re.S)
    return blocks + re.findall(r"`([^`\n]+)`", without_blocks)


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
    plugin = json.loads((_plugin_dir() / ".claude-plugin" / "plugin.json").read_text())
    assert plugin["name"] == REPO
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert plugin["version"] == project["version"]


def test_skill_frontmatter_is_portable_and_small():
    text = (_skill_dir() / "SKILL.md").read_text()
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
        for target in re.findall(r"\]\(([^)\s#]+)(?:#[^)]*)?\)", path.read_text()):
            if re.match(r"[a-z]+:", target):
                continue
            assert (path.parent / target).exists(), f"{path.name}: broken link {target}"


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


def _cli_root():
    """The click command tree. Duck-typed: typer may bring its own copy of click."""
    module, _, attr = CLI_ENTRY.partition(":")
    obj = getattr(importlib.import_module(module), attr)
    if hasattr(obj, "params"):
        return obj
    return typer.main.get_command(obj)


def _invocations(text: str) -> list[list[str]]:
    """Token lists after the program name, where the program is the command being run: at the
    start of a line, after a shell separator, or after `python -m`."""
    start = r"(?:^|&&|\|\||[;|(]|\$\(|python3? -m)\s*"
    out = []
    for span in _code_spans(text):
        for line in span.splitlines():
            line = line.split(" #", 1)[0].strip()
            for m in re.finditer(rf"{start}{re.escape(CLI_PROG)}((?:[ \t]+[^\s;&|)]+)*)", line):
                out.append(m.group(1).split())
    return out


def _options(cmd) -> set[str]:
    return {o for p in cmd.params for o in (*p.opts, *p.secondary_opts)} | {"--help"}


def test_skill_cli_invocations_exist():
    root = _cli_root()
    calls = _invocations(_skill_text())
    assert calls, f"the skill should show {CLI_PROG} commands"
    bad = []
    for tokens in calls:
        cmd = root
        for tok in tokens:
            if tok.startswith("<") or tok in {"...", "…"}:
                continue
            if tok.startswith("--"):
                opt = tok.split("=", 1)[0]
                if opt not in _options(cmd):
                    bad.append(f"{' '.join(tokens)}: {opt}")
            elif not tok.startswith("-") and hasattr(cmd, "commands"):
                if tok not in cmd.commands:
                    bad.append(f"{' '.join(tokens)}: no subcommand {tok}")
                    break
                cmd = cmd.commands[tok]
    assert not bad, f"unknown CLI commands/options: {bad}"
