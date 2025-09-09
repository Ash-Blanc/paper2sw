"""Pytest configuration file."""

import sys
from pathlib import Path

# Add the src directory to the path so we can import the package
src_path = Path(__file__).parent / "paper2sw" / "src"
sys.path.insert(0, str(src_path))