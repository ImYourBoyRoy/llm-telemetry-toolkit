# ./tests/test_helper.py
"""
Provide reusable setup utilities for this test suite.
Used by test modules to enforce clean import state and source-path resolution.
Run: Imported by `unittest` or `pytest` test modules.
Inputs: Project root path inferred from file location.
Outputs: Cleaned `__pycache__` directories and configured `sys.path`.
Side effects: Deletes cache directories under project root during test startup.
Operational notes: Cache scrubbing guarantees fresh bytecode for continuity checks.
"""

import sys
import shutil
from pathlib import Path


def clean_pycache(root_dir: Path):
    """
    Recursively deletes __pycache__ directories within the given root directory.
    """
    for pycache in root_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            print(f"Cleaned: {pycache}")
        except Exception as e:
            print(f"Failed to clean {pycache}: {e}")


def setup_test_environment():
    """
    Standard setup for all test scripts.
    Cleans pycache and ensures src is in path.
    """
    root_dir = Path(__file__).parent.parent
    clean_pycache(root_dir)

    src_path = root_dir / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
