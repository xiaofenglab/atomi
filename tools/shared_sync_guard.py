#!/usr/bin/env python3
"""Shared-folder guard for editing Atomi from multiple sessions.

This does not prove Google Drive has uploaded every byte to every device.
It gives editing sessions a practical protocol:

- acquire: write ATOMI_EDIT_LOCK.json before editing
- release: mark clean/synced and remove the lock after pushing or pausing
- status: show whether the folder is safe to edit

The flag files live in the repository root so Google Drive syncs them, but
they are ignored by Git.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path


LOCK_FILE = "ATOMI_EDIT_LOCK.json"
STATUS_FILE = "ATOMI_SYNC_STATUS.json"
PROBE_FILE = "ATOMI_SYNC_PROBE.txt"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return Path.cwd()
    return Path(out)


def git_text(args: list[str], root: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ""


def git_clean(root: Path) -> bool:
    return git_text(["status", "--short"], root) == ""


def git_head(root: Path) -> str:
    return git_text(["rev-parse", "--short", "HEAD"], root)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"invalid_json": path.read_text(errors="replace")}


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def identity() -> dict:
    return {
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "platform": platform.platform(),
    }


def write_probe(root: Path, wait: float) -> dict:
    probe = root / PROBE_FILE
    payload = {
        "timestamp": now(),
        "identity": identity(),
        "message": "Google Drive sync probe. Safe to delete.",
    }
    probe.write_text(json.dumps(payload, sort_keys=True) + "\n")
    first = probe.stat()
    time.sleep(wait)
    second = probe.stat()
    stable = first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns
    return {"path": str(probe), "stable_for_seconds": wait, "locally_stable": stable}


def status(root: Path, probe_wait: float = 0.0) -> int:
    lock = read_json(root / LOCK_FILE)
    status_data = read_json(root / STATUS_FILE)
    clean = git_clean(root)
    head = git_head(root)

    probe = None
    if probe_wait > 0:
        probe = write_probe(root, probe_wait)

    safe = clean and not lock
    result = {
        "repo": str(root),
        "safe_to_edit": safe,
        "git_clean": clean,
        "git_head": head,
        "lock": lock or None,
        "sync_status": status_data or None,
        "probe": probe,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if safe else 2


def acquire(root: Path, note: str, force: bool = False) -> int:
    lock_path = root / LOCK_FILE
    existing = read_json(lock_path)
    if existing and not force:
        print(f"Refusing to acquire: {LOCK_FILE} already exists.", file=sys.stderr)
        print(json.dumps(existing, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    data = {
        "state": "editing",
        "created_at": now(),
        "note": note,
        "git_head": git_head(root),
        "identity": identity(),
    }
    write_json(lock_path, data)
    write_json(
        root / STATUS_FILE,
        {
            "state": "editing",
            "updated_at": now(),
            "git_head": data["git_head"],
            "note": note,
            "identity": data["identity"],
        },
    )
    print(f"Acquired edit lock: {lock_path}")
    return 0


def release(root: Path, note: str, keep_lock: bool = False) -> int:
    clean = git_clean(root)
    data = {
        "state": "synced" if clean else "dirty",
        "updated_at": now(),
        "git_clean": clean,
        "git_head": git_head(root),
        "note": note,
        "identity": identity(),
    }
    write_json(root / STATUS_FILE, data)
    if clean and not keep_lock:
        lock_path = root / LOCK_FILE
        if lock_path.exists():
            lock_path.unlink()
    print(json.dumps(data, indent=2, sort_keys=True))
    if not clean:
        print("Repository is dirty; keeping lock until changes are committed/stashed/cleaned.", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coordinate multi-session editing in a Google Drive repo copy.")
    parser.add_argument("--root", type=Path, default=None, help="Repository root. Default: git top-level or cwd.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show lock, git, and sync flag status.")
    p_status.add_argument("--probe-wait", type=float, default=0.0, help="Write a probe file and confirm local stability after N seconds.")

    p_acquire = sub.add_parser("acquire", help="Acquire edit lock before changing files.")
    p_acquire.add_argument("--note", default="", help="Short description of planned work.")
    p_acquire.add_argument("--force", action="store_true", help="Overwrite an existing stale lock.")

    p_release = sub.add_parser("release", help="Mark synced and remove lock if git is clean.")
    p_release.add_argument("--note", default="", help="Short completion note.")
    p_release.add_argument("--keep-lock", action="store_true", help="Keep lock even if git is clean.")

    args = parser.parse_args(argv)
    root = args.root.resolve() if args.root else repo_root()

    if args.command == "status":
        return status(root, probe_wait=args.probe_wait)
    if args.command == "acquire":
        return acquire(root, note=args.note, force=args.force)
    if args.command == "release":
        return release(root, note=args.note, keep_lock=args.keep_lock)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
