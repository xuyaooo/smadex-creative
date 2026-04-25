"""
Top-level pytest configuration.

Adds the project root to sys.path so tests can `import src.*` modules without
needing an installed package, mirroring how scripts/eval.py and demo run.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
