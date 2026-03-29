# Reference App Overview

The Streamlit sandbox at `apps/sandbox/` is the reference implementation for
Engram's full feature set.

It demonstrates:
- The complete five-layer memory system in a real chat loop
- Token budget management and pressure valve behavior
- The forgetting policy lifecycle (episodes → cold storage)
- Engine failover and runtime management
- Live inspection of retrieval, context assembly, and diagnostics

See `docs/architecture/reference-app.md` for a description of each tab.
See `docs/tutorials/build-your-own-app.md` for building your own UI on top
of the same `ProjectMemory` interface.
