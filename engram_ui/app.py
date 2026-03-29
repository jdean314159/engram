"""Compatibility launcher for the Engram sandbox reference app.

Preferred entry point:
    streamlit run apps/sandbox/app.py

This file remains as a compatibility shim for older instructions.
"""

from apps.sandbox.app import main


if __name__ == "__main__":
    main()
