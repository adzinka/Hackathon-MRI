"""Fetch (or verify) the real MRI samples listed in external_data/real_samples/dataset_manifest.json.

Only the listed members are read from the 1.68 GB Zenodo archive, with HTTP range requests; zipfile checks each
member's CRC-32 and this script checks size and SHA-256 against the manifest before writing. A file is written only
when it is complete and its checksum matches, so a failed or partial download never lands in the repository.

    python scripts/fetch_real_samples.py            # download what is missing or wrong, then verify everything
    python scripts/fetch_real_samples.py --check    # verify only (exit code 1 if anything is missing or wrong)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "external_data" / "real_samples" / "dataset_manifest.json"


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file; each read is one HTTP Range request."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            if r.status != 206:
                raise IOError(f"server ignored the Range request (HTTP {r.status})")
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ok(entry: dict) -> bool:
    p = REPO / entry["repo_path"]
    return p.exists() and p.stat().st_size == entry["size_bytes"] and sha256(p) == entry["sha256"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify only, do not download")
    args = ap.parse_args()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    todo = [e for e in man["files"] if not ok(e)]
    if todo and not args.check:
        url = man["dataset"]["archive"]["download_url"]
        print(f"reading {len(todo)} member(s) from {url}")
        z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=4 << 20))
        for e in todo:
            data = z.read(e["archive_member"])                       # BadZipFile on CRC mismatch
            got = hashlib.sha256(data).hexdigest()
            if len(data) != e["size_bytes"] or got != e["sha256"]:
                print(f"  MISMATCH {e['archive_member']}: {len(data)} B, sha256 {got}; not written")
                continue
            dest = REPO / e["repo_path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            print(f"  wrote {e['repo_path']} ({len(data):,} B)")
    bad = 0
    for e in man["files"]:
        good = ok(e)
        bad += not good
        print(f"{'OK     ' if good else 'MISSING'} {e['repo_path']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
