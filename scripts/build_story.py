"""Build the ScanTwin pitch page.

Inputs (edit these, not the output):
  docs/story_src.html            the deck: slides, styles, motion, speaker notes
  docs/story_assets/data.json    zoomed Cho/Cr spectra (StarterPack MRS, 4x zero-filled), T1w slices and region
                                 outlines, two simulated scanner-twin phantom records (shim degrading, stable)
  docs/story_assets/meshes.json  brain, CSF and region surfaces from the StarterPack NIfTI (mm, RAS)

Outputs:
  docs/story.html                self-contained (meshes inline). Open it on the pitch laptop; works offline except
                                 for the Three.js and font CDNs, which fall back gracefully.
  --publish OUT.html             data inline, meshes left out: the page fetches story_meshes.json published beside it.

Usage:
  python scripts/build_story.py [--publish /path/to/out.html]
"""
from __future__ import annotations

import argparse
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _safe(js: str) -> str:
    """JSON placed inside <script> must not contain a closing tag."""
    return js.replace("</", "<\\/")


def build(publish: Path | None) -> None:
    src = (DOCS / "story_src.html").read_text(encoding="utf-8")
    data = _safe((DOCS / "story_assets" / "data.json").read_text(encoding="utf-8").strip())
    meshes = _safe((DOCS / "story_assets" / "meshes.json").read_text(encoding="utf-8").strip())
    for token in ("__DATA__", "__MESHES__"):
        if src.count(token) != 1:
            raise SystemExit(f"{token} must appear exactly once in story_src.html")
    page = src.replace("__DATA__", data, 1)
    (DOCS / "story.html").write_text(page.replace("__MESHES__", meshes, 1), encoding="utf-8")
    print("wrote", DOCS / "story.html")
    if publish:
        publish.write_text(page.replace("__MESHES__", "", 1), encoding="utf-8")
        print("wrote", publish, "(publish story_assets/meshes.json beside it as story_meshes.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publish", type=Path, help="also write a page without inline meshes, for publishing")
    build(ap.parse_args().publish)
