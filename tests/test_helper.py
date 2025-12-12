# ./llm-telemetry-toolkit/tests/test_helper.py
"""
Test Helper Module.
Provides utilities for test setup, specifically __pycache__ cleaning.
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
