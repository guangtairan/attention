#!/usr/bin/env python
"""Clean hidden characters in configs/*.py and optionally validate syntax.

Usage:
  python scripts/clean_configs_hidden_chars.py
  python scripts/clean_configs_hidden_chars.py --root /root/autodl-tmp/mmsegmentation-sci/configs
"""

from __future__ import annotations

import argparse
from pathlib import Path


BAD_CHARS = [
    "\ufeff",  # BOM / ZWNBSP
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\xa0",    # NO-BREAK SPACE
]


def clean_file(path: Path) -> bool:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")  # auto-strip BOM
    except UnicodeDecodeError:
        return False

    cleaned = text
    for ch in BAD_CHARS:
        cleaned = cleaned.replace(ch, "")

    changed = cleaned != text or raw.startswith(b"\xef\xbb\xbf")
    if changed:
        path.write_text(cleaned, encoding="utf-8", newline="\n")
    return changed


def check_python_syntax(path: Path) -> tuple[bool, str]:
    try:
        src = path.read_text(encoding="utf-8")
        compile(src, str(path), "exec")
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="configs",
        help="Root folder to scan (default: configs)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERROR] Not found: {root}")
        return 2

    py_files = sorted(root.rglob("*.py"))
    changed_files = []
    for p in py_files:
        if clean_file(p):
            changed_files.append(p)
            print(f"[FIXED] {p}")

    print(f"[DONE] cleaned {len(changed_files)} / {len(py_files)} files")

    syntax_failed = []
    for p in py_files:
        ok, msg = check_python_syntax(p)
        if not ok:
            syntax_failed.append((p, msg))

    if syntax_failed:
        print("[SYNTAX] failed files:")
        for p, msg in syntax_failed:
            print(f"  - {p}: {msg}")
        return 1

    print("[SYNTAX] all .py files under configs are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
