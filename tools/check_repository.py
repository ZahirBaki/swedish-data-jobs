"""Scan publishable files and optional Git history without revealing secrets.

Run ``python -m tools.check_repository --scan-history`` before staging changes.
This is a local heuristic check, not a guarantee that every secret is detected.
"""

import argparse
import ast
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "provider_token": re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "literal_credential": re.compile(
        rb"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[=:]\s*[\"']([A-Za-z0-9_+/.=-]{20,})[\"']"
    ),
}
SENSITIVE = re.compile(r"(?i)(api.?key|secret|password|access.?token)")
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]|/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/)")


def git(*args):
    """Capture Git output; never print blob contents or remote credentials."""
    return subprocess.check_output(["git", *args], cwd=ROOT)


def known_local_secrets():
    """Use exact local values as additional detectors, never as report content."""
    values = {v.encode() for k, v in os.environ.items() if SENSITIVE.search(k) and len(v) >= 16}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.partition("=")
            value = value.strip().strip("\"'")
            if separator and SENSITIVE.search(key) and len(value) >= 16:
                values.add(value.encode())
    return values


def secret_kinds(content, known=()):
    kinds = [name for name, pattern in PATTERNS.items() if pattern.search(content)]
    if any(value in content for value in known):
        kinds.append("exact_local_credential")
    return kinds


def notebook_issues(content):
    notebook = json.loads(content)
    return any(cell.get("outputs") or cell.get("execution_count") is not None
               for cell in notebook.get("cells", []) if cell.get("cell_type") == "code")


def contains_home_path(value):
    """Inspect decoded notebook strings, including escaped output paths."""
    if isinstance(value, str):
        return bool(LOCAL_PATH.search(value))
    if isinstance(value, dict):
        return any(contains_home_path(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_home_path(item) for item in value)
    return False


def inspect(history=False):
    """Inspect tracked and unignored candidates; ignored local data is not published."""
    known = known_local_secrets()
    paths = sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""})
    findings, syntax_errors, notebook_outputs, local_paths, oversized = [], [], [], [], []
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            continue  # Deleted files appear in the index until staged.
        content = path.read_bytes()
        for kind in secret_kinds(content, known):
            findings.append({"path": name, "kind": kind, "location": "working_tree"})
        if len(content) > 2_000_000:
            oversized.append(name)
        if path.suffix in {".py", ".md", ".ipynb", ".toml", ".yml", ".yaml"}:
            text = content.decode("utf-8-sig")
            if contains_home_path(json.loads(text) if path.suffix == '.ipynb' else text):
                local_paths.append(name)
            if path.suffix == ".py":
                try:
                    ast.parse(text, filename=name)
                except SyntaxError:
                    syntax_errors.append(name)
            if path.suffix == ".ipynb" and notebook_issues(text):
                notebook_outputs.append(name)
    historical_blobs = 0
    historical_notebook_outputs, historical_home_paths = [], []
    if history:
        for entry in git("rev-list", "--objects", "--all").decode().splitlines():
            oid, _, name = entry.partition(" ")
            if git("cat-file", "-t", oid).strip() != b"blob":
                continue
            historical_blobs += 1
            content = git("cat-file", "blob", oid)
            for kind in secret_kinds(content, known):
                findings.append({"path": name, "object_id": oid, "kind": kind, "location": "git_history"})
            if name.endswith('.ipynb'):
                try:
                    notebook = json.loads(content)
                    if notebook_issues(content):
                        historical_notebook_outputs.append({"path": name, "object_id": oid})
                    if contains_home_path(notebook):
                        historical_home_paths.append({"path": name, "object_id": oid})
                except (ValueError, UnicodeError):
                    pass
    return {
        "candidate_files": len(paths), "history_scanned": history,
        "historical_blobs_scanned": historical_blobs, "secret_findings": findings,
        "syntax_errors": syntax_errors, "notebooks_with_outputs": notebook_outputs,
        "hardcoded_home_paths": local_paths, "files_over_2mb": oversized,
        "historical_notebook_outputs": historical_notebook_outputs,
        "historical_notebook_home_paths": historical_home_paths,
        "scope": "Tracked files, unignored candidates and optionally blobs reachable from all local Git refs; no network or secret values printed.",
        "limitations": "Heuristic patterns and exact local credential matches cannot prove the absence of every secret; remote-only refs and unreachable objects are not scanned.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-history", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = inspect(args.scan_history)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return int(any(report[key] for key in ["secret_findings", "syntax_errors", "notebooks_with_outputs", "hardcoded_home_paths", "files_over_2mb"]))


if __name__ == "__main__":
    raise SystemExit(main())
