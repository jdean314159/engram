"""
pytest conftest — makes all harness tests discoverable by pytest.

The harness uses its own runner (run_tests.py) but every test function
also works as a plain pytest test: they raise on failure, return None on
pass, and raise SkipTest (which pytest recognises) on skip.

Run with:
    PYTHONPATH=. pytest tests/harness/ -v
    PYTHONPATH=. pytest tests/harness/ -v -k "Working Memory"
    PYTHONPATH=. pytest tests/harness/ -v --tb=short
"""
import sys
from pathlib import Path

# Raise the open-file-descriptor limit before any tests run.
# Kuzu opens multiple file handles per database; the default limit (1024)
# is easily exhausted when many tests with SemanticMemory/ProjectMemory run
# sequentially.
try:
    import resource as _resource
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _target = min(max(_soft, 4096), _hard) if _hard > 0 else max(_soft, 4096)
    if _target > _soft:
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))
except Exception:
    pass

# Ensure project root is on sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
