from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True, kw_only=True, frozen=True)
class ManifestEntry:
    relpath: str
    bytes: int
    sha256: str


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def write_manifest(root: Path, glob_pattern: str = "**/*", out: Path | None = None) -> Path:
    root = root.resolve()
    entries: list[ManifestEntry] = []
    for p in sorted(root.glob(glob_pattern)):
        if not p.is_file():
            continue
        entries.append(
            ManifestEntry(
                relpath=str(p.relative_to(root)),
                bytes=p.stat().st_size,
                sha256=sha256_file(p),
            )
        )
    out = out or (root / "manifest.json")
    out.write_text(json.dumps([asdict(e) for e in entries], indent=2))
    return out


def manifest_digest(manifest_path: Path) -> str:
    return sha256_file(manifest_path)
