"""CoderAI - a real coding agent CLI (like a mini OpenCode/ChatGPT-for-code).

Uses the OpenAI API (or any OpenAI-compatible endpoint) with tool calling.
Tools: read_file, write_file, edit_file, list_dir, run_command, search.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

MODEL = os.environ.get("CODERAI_MODEL", os.environ.get("OPENAI_MODEL", "openai"))
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://text.pollinations.ai/openai")  # free, no key needed
API_KEY = os.environ.get("OPENAI_API_KEY", "no-key-needed")

SYSTEM = """You are CoderAI, an autonomous coding assistant in a terminal.
You can read, write and edit files, list directories, search code and run shell commands.
Work step by step: explore first, make small edits, verify by running commands.
Keep explanations concise. When done, summarize what you changed."""

# ---------------- Tools ----------------

def run_command(command: str, timeout: int = 60) -> str:
    try:
        p = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout,
            cwd=os.getcwd(),
        )
        out = (p.stdout or "") + (p.stderr or "")
        return out[-8000:] if out else f"(exit code {p.returncode}, no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str) -> str:
    try:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > 20000:
            text = text[:20000] + "\n... (truncated)"
        return text
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    try:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if old_string not in text:
            return "Error: old_string not found in file"
        if text.count(old_string) > 1:
            return "Error: old_string occurs multiple times; make it unique"
        p.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def list_dir(path: str = ".") -> str:
    try:
        entries = []
        for e in sorted(Path(path).iterdir()):
            entries.append(f"{'[dir] ' if e.is_dir() else ''}{e.name}")
        return "\n".join(entries) or "(empty)"
    except Exception as e:
        return f"Error: {e}"


def search(pattern: str, path: str = ".") -> str:
    try:
        import re
        hits = []
        for p in Path(path).rglob("*"):
            if p.is_file() and p.stat().st_size < 2_000_000:
                try:
                    for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if re.search(pattern, line):
                            hits.append(f"{p}:{i}: {line.strip()[:120]}")
                            if len(hits) >= 50:
                                return "\n".join(hits) + "\n... (truncated)"
                except Exception:
                    pass
        return "\n".join(hits) or "(no matches)"
    except Exception as e:
        return f"Error: {e}"


TOOL_FNS = {
    "run_command": lambda a: run_command(a["command"], int(a.get("timeout", 60))),
    "read_file": lambda a: read_file(a["path"]),
    "write_file": lambda a: write_file(a["path"], a["content"]),
    "edit_file": lambda a: edit_file(a["path"], a["old_string"], a["new_string"]),
    "list_dir": lambda a: list_dir(a.get("path", ".")),
    "search": lambda a: search(a["pattern"], a.get("path", ".")),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace a unique substring in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_string": {"type": "string"},
            "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List files in a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Regex search across files.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"]}}},
]

# ---------------- Agent loop ----------------

def run_agent_turn(client: OpenAI, messages: list, user_input: str) -> None:
    messages.append({"role": "user", "content": user_input})
    while True:
        with console.status("[bold green]Thinking..."):
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, temperature=0.2,
            )
        msg = resp.choices[0].message
        messages.append(msg)

        if msg.content:
            console.print(Markdown(msg.content))

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            console.print(Panel(f"[bold cyan]{name}[/] {json.dumps(args)[:200]}", title="tool call", border_style="cyan"))
            fn = TOOL_FNS.get(name)
            result = fn(args) if fn else f"Unknown tool: {name}"
            shown = result if len(result) < 1500 else result[:1500] + "\n... (truncated)"
            console.print(Panel(shown, title="result", border_style="yellow"))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


def main() -> None:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "system", "content": SYSTEM}]
    console.print(Panel(f"CoderAI — model [bold]{MODEL}[/]\nType your task, /exit to quit, /clear to reset.", title="CoderAI", border_style="green"))
    while True:
        try:
            user_input = console.input("[bold blue]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/clear":
            messages = [{"role": "system", "content": SYSTEM}]
            console.print("[dim]conversation cleared[/dim]")
            continue
        try:
            run_agent_turn(client, messages, user_input)
        except Exception as e:
            console.print(f"[red]{e}[/red]")


if __name__ == "__main__":
    main()
