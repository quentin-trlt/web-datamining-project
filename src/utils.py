"""Shared utility: resolve paths relative to the project root (parent of src/)."""

from pathlib import Path

# Project root = parent of the directory containing this file (src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(relative: str) -> str:
    """Resolve a path relative to the project root, e.g. 'kg_artifacts/family.owl'."""
    return str(PROJECT_ROOT / relative)
