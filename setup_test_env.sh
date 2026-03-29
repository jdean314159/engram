#!/usr/bin/env bash
# =============================================================================
# Engram Full Test Environment Setup
#
# Installs all optional dependencies needed to run the complete test suite,
# including tests that currently skip due to missing: chromadb, kuzu, torch,
# sentence-transformers, diskcache, pydantic, openai, sklearn.
#
# Usage:
#   cd ~/ai_tools/engram          # or wherever you extracted the tarball
#   bash setup_test_env.sh
#
# Tested on: Ubuntu 24.04, Python 3.11/3.12, RTX 3090 (CUDA 12.x)
#
# After setup, run the full suite with:
#   PYTHONPATH=. python run_tests.py
#   # or
#   pytest tests/harness/ -v
# =============================================================================

set -euo pipefail

PYTHON=${PYTHON:-python3}
PIP="$PYTHON -m pip"

echo "=== Engram test environment setup ==="
echo "Python: $($PYTHON --version)"
echo ""

# --- Detect pip install flag ---
# Ubuntu 24.04 with system Python requires --break-system-packages
if $PYTHON -c "import sys; sys.exit(0 if sys.prefix == sys.base_prefix else 1)" 2>/dev/null; then
    # Not in a venv — need the flag
    PIP_FLAGS="--break-system-packages"
    echo "NOTE: Not in a virtualenv — using --break-system-packages"
    echo "      Consider: python3 -m venv .venv && source .venv/bin/activate"
else
    PIP_FLAGS=""
    echo "Virtualenv detected — no flags needed"
fi
echo ""

# --- Helper ---
pip_install() {
    $PIP install $PIP_FLAGS "$@"
}

# --- Core (required by all tests) ---
echo "=== [1/7] Core dependencies ==="
pip_install pydantic>=2.0.0 numpy>=1.24.0 pyyaml>=6.0.0

# --- Engine extras ---
echo ""
echo "=== [2/7] Engine extras (openai, tiktoken, json-repair) ==="
pip_install openai>=1.0.0 tiktoken>=0.5.0 "json-repair>=0.7.0"
pip_install anthropic>=0.18.0 tenacity>=8.0.0

# --- Episodic memory (chromadb + sentence-transformers) ---
echo ""
echo "=== [3/7] Episodic memory (chromadb + sentence-transformers) ==="
echo "    NOTE: chromadb pulls ~500MB of dependencies including onnxruntime."
echo "    sentence-transformers will download all-MiniLM-L6-v2 on first use (~90MB)."
pip_install "chromadb>=0.4.22" "sentence-transformers>=2.2.0" "diskcache>=5.6.2"

# --- Semantic memory (kuzu graph DB) ---
echo ""
echo "=== [4/7] Semantic memory (kuzu) ==="
pip_install "kuzu>=0.4.0"

# --- Graph extraction (scikit-learn — usually already present) ---
echo ""
echo "=== [5/7] Graph extraction (scikit-learn) ==="
pip_install "scikit-learn>=1.0.0"

# --- Neural memory (torch) ---
echo ""
echo "=== [6/7] Neural memory (torch) ==="
echo "    NOTE: If you have CUDA 12.x installed, torch will use the GPU automatically."
echo "    This downloads ~2GB. Skipping if already installed."
if ! $PYTHON -c "import torch" 2>/dev/null; then
    # Detect CUDA
    if command -v nvcc &>/dev/null || nvidia-smi &>/dev/null 2>&1; then
        CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
        echo "    CUDA detected (driver: $CUDA_VER) — installing torch with CUDA support"
        pip_install torch --index-url https://download.pytorch.org/whl/cu121
    else
        echo "    No CUDA detected — installing CPU-only torch"
        pip_install torch --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "    torch already installed: $($PYTHON -c 'import torch; print(torch.__version__)')"
fi

# --- Test tooling ---
echo ""
echo "=== [7/7] Test tooling ==="
pip_install "pytest>=7.4.0" "pytest-asyncio>=0.21.0"

# --- Install engram itself in editable mode ---
echo ""
echo "=== Installing engram in editable mode ==="
pip_install -e ".[dev]" || pip_install -e . 

# --- Verify ---
echo ""
echo "=== Dependency verification ==="
$PYTHON - << 'PYEOF'
import sys
deps = {
    "pydantic":              "pydantic",
    "numpy":                 "numpy",
    "chromadb":              "chromadb",
    "sentence_transformers": "sentence_transformers",
    "diskcache":             "diskcache",
    "kuzu":                  "kuzu",
    "torch":                 "torch",
    "sklearn":               "sklearn",
    "openai":                "openai",
    "anthropic":             "anthropic",
    "tiktoken":              "tiktoken",
    "pytest":                "pytest",
}
ok = []
missing = []
for name, mod in deps.items():
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        ok.append(f"  ✓ {name:<25} {ver}")
    except ImportError:
        missing.append(f"  ✗ {name}")

print("Available:")
print("\n".join(ok))
if missing:
    print("\nMissing (some tests will skip):")
    print("\n".join(missing))
    sys.exit(0)  # not fatal — optional deps may be intentionally absent
else:
    print("\nAll dependencies available!")
PYEOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run the full test suite:"
echo "  cd $(pwd)"
echo "  PYTHONPATH=. python run_tests.py"
echo ""
echo "Run with pytest:"
echo "  PYTHONPATH=. pytest tests/harness/ -v"
echo ""
echo "Run a specific group:"
echo "  PYTHONPATH=. python run_tests.py --groups 'Episodic Memory'"
echo ""
echo "Skip slow deps (torch/chromadb tests):"
echo "  PYTHONPATH=. python run_tests.py --fast"
