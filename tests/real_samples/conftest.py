"""Shared helpers for the real-data tests (Morelli et al., Zenodo 10.5281/zenodo.13753316, sub-01, CC BY 4.0)."""
import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "external_data" / "real_samples"
MANIFEST = json.loads((DATA / "dataset_manifest.json").read_text(encoding="utf-8"))
REPORT = DATA / "qc_report.json"


def entry(name: str) -> dict:
    return next(e for e in MANIFEST["files"] if e["repo_path"].endswith(name))


@pytest.fixture(scope="session")
def real_file():
    """Path of a manifest file, after checking it is complete and unmodified (skip if it was never fetched)."""
    def get(name: str) -> Path:
        e = entry(name)
        p = REPO / e["repo_path"]
        if not p.exists():
            pytest.skip(f"{e['repo_path']} missing: run  python scripts/fetch_real_samples.py")
        assert p.stat().st_size == e["size_bytes"], f"{p.name}: size differs from the manifest (partial download?)"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == e["sha256"], f"{p.name}: SHA-256 differs from the manifest"
        return p
    return get


@pytest.fixture(scope="session")
def report():
    return json.loads(REPORT.read_text(encoding="utf-8"))
