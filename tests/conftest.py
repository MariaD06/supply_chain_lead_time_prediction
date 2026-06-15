# conftest.py is intentionally empty.
# The project is installed as a package via `pip install -e .`
# which makes `import src...` work in tests without sys.path manipulation.




#import sys
#from pathlib import Path

# Ensure project root is on sys.path so `import src...` works in tests
#ROOT = Path(__file__).resolve().parents[1]
#if str(ROOT) not in sys.path:
#    sys.path.insert(0, str(ROOT))
