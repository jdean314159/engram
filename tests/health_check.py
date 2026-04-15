def health_check(self) -> Dict[str, Any]:
    """Validate all memory layers are functional.

    Runs a lightweight read/write probe on each layer.
    Raises RuntimeError if any critical layer fails.
    Logs WARNING for non-critical failures.

    Returns a report dict suitable for display in the UI or logs.
    Can be called at any time, not just at init.
    """
    report = {
        "project_id": self.project_id,
        "timestamp": time.time(),
        "layers": {},
        "warnings": [],
        "critical": [],
    }

    # --- Working memory (critical) ---
    try:
        msg = self.working.add("__health__", "__probe__")
        self.working.clear_session()
        self.working.add  # re-init session
        report["layers"]["working"] = "ok"
    except Exception as exc:
        report["layers"]["working"] = f"FAILED: {exc}"
        report["critical"].append(f"Working memory: {exc}")

    # --- Episodic memory (warning) ---
    if self.episodic is None:
        report["layers"]["episodic"] = "disabled"
    else:
        try:
            self.episodic.get_recent_episodes(n=1, project_id=self.project_id)
            report["layers"]["episodic"] = "ok"
        except Exception as exc:
            report["layers"]["episodic"] = f"FAILED: {exc}"
            report["warnings"].append(f"Episodic memory: {exc}")
            logger.warning("HEALTH CHECK: Episodic memory failed: %s", exc)

    # --- Semantic memory (loud warning) ---
    if self.semantic is None:
        report["layers"]["semantic"] = "disabled"
        report["warnings"].append("Semantic memory is disabled — check initialization logs")
        logger.warning("HEALTH CHECK: Semantic memory is disabled")
    else:
        try:
            probe_id = f"__health_probe_{int(time.time())}"
            self.semantic.add_fact(
                "__health_probe__",
                confidence=0.0,
                source="health_check",
                fact_id=probe_id,
            )
            facts = self.semantic.list_facts(limit=1)
            self.semantic.delete_node("Fact", probe_id)
            report["layers"]["semantic"] = "ok"
            report["layers"]["semantic_backend"] = getattr(self.semantic, "_db_file", "unknown")
        except Exception as exc:
            report["layers"]["semantic"] = f"FAILED: {exc}"
            report["warnings"].append(f"Semantic memory: {exc}")
            logger.warning("HEALTH CHECK: Semantic memory failed probe: %s", exc)

    # --- Daemon (critical) ---
    if not hasattr(self, "_daemon"):
        report["layers"]["daemon"] = "not started"
        report["critical"].append("MemoryDaemon not started")
    elif not self._daemon._thread.is_alive():
        report["layers"]["daemon"] = "DEAD"
        report["critical"].append("MemoryDaemon thread is not alive")
    else:
        report["layers"]["daemon"] = "ok"
        report["layers"]["daemon_stats"] = self._daemon.get_stats()

    # --- Path validation (warning) ---
    for name, path in [
        ("project_dir", self._project_dir),
        ("working_db", self._project_dir / "working.db"),
    ]:
        if not Path(str(path)).is_absolute():
            msg = f"Path not absolute: {name}={path}"
            report["warnings"].append(msg)
            logger.warning("HEALTH CHECK: %s", msg)

    # --- Cognitive layer (warning) ---
    if hasattr(self, "_daemon"):
        cog_stats = self._daemon._cognitive.get_stats()
        if not cog_stats.get("enabled"):
            report["layers"]["cognitive"] = "disabled"
        else:
            report["layers"]["cognitive"] = "ok"

    # --- Raise on critical failures ---
    if report["critical"]:
        raise RuntimeError(
            f"ProjectMemory health check failed for {self.project_id}: "
            + "; ".join(report["critical"])
        )

    # Log summary
    warning_count = len(report["warnings"])
    if warning_count:
        logger.warning(
            "HEALTH CHECK: project=%s passed with %d warning(s): %s",
            self.project_id, warning_count, "; ".join(report["warnings"])
        )
    else:
        logger.info("HEALTH CHECK: project=%s all layers ok", self.project_id)

    return report
