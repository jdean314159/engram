from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

def _runtime_memory_config_snapshot(pm) -> Dict[str, Any]:
    result = {}

    token_budget = getattr(pm, "budget", None)
    if token_budget is not None:
        result["Working budget"] = getattr(token_budget, "working", None)
        result["Episodic budget"] = getattr(token_budget, "episodic", None)
        result["Semantic budget"] = getattr(token_budget, "semantic", None)
        result["Cold budget"] = getattr(token_budget, "cold", None)
        result["Total budget"] = getattr(token_budget, "total", None)

    result["Project ID"] = getattr(pm, "project_id", None)
    result["Project type"] = getattr(pm, "project_type", None)
    result["Session ID"] = getattr(pm, "session_id", None)

    llm_engine = getattr(pm, "llm_engine", None)
    if llm_engine is not None:
        result["Engine"] = getattr(llm_engine, "name", None) or llm_engine.__class__.__name__

    emb_cache = getattr(pm, "embedding_cache", None)
    if emb_cache is not None:
        try:
            cs = emb_cache.get_stats()
            if cs.get("enabled"):
                hit_rate = cs.get("hit_rate", 0.0)
                result["Embed cache hit rate"] = f"{hit_rate:.1%}"
                result["Embed cache size"] = cs.get("memory_size", 0)
                result["Embed cache disk"] = "yes" if cs.get("has_disk") else "no"
        except Exception:
            pass

    return {k: v for k, v in result.items() if v is not None}


def _render_summary_metrics(title: str, metrics: Dict[str, Any]) -> None:
    st.markdown(f"**{title}**")
    if not metrics:
        st.caption("No summary available.")
        return

    cols = st.columns(max(1, min(4, len(metrics))))
    for idx, (k, v) in enumerate(metrics.items()):
        cols[idx % len(cols)].metric(k, str(v))


def _render_record_list(
    records: List[Dict[str, Any]],
    title: str,
    empty_text: str = "No records found.",
) -> None:
    st.markdown(f"**{title}**")
    if not records:
        st.caption(empty_text)
        return

    for idx, rec in enumerate(records):
        label = rec.get("title") or rec.get("id") or f"Item {idx + 1}"
        with st.expander(str(label), expanded=False):
            st.json(rec)
            
def _normalize_message(item: Any, idx: int) -> Dict[str, Any]:
    if isinstance(item, dict):
        rec = dict(item)
    else:
        rec = {
            "role": getattr(item, "role", None),
            "content": getattr(item, "content", None),
            "timestamp": getattr(item, "timestamp", None),
        }

    title_bits = []
    if rec.get("timestamp"):
        title_bits.append(str(rec["timestamp"]))
    if rec.get("role"):
        title_bits.append(str(rec["role"]))
    rec["title"] = " | ".join(title_bits) if title_bits else f"Turn {idx + 1}"
    return rec            
    
def _normalize_episode(item: Any, idx: int) -> Dict[str, Any]:
    if isinstance(item, dict):
        rec = dict(item)
    else:
        rec = {
            "id": getattr(item, "id", idx),
            "text": getattr(item, "text", None),
            "score": getattr(item, "score", None),
            "timestamp": getattr(item, "timestamp", None),
            "metadata": getattr(item, "metadata", None),
        }

    title_bits = []
    if rec.get("timestamp"):
        title_bits.append(str(rec["timestamp"]))
    if rec.get("id"):
        title_bits.append(str(rec["id"]))
    rec["title"] = " | ".join(title_bits) if title_bits else f"Episode {idx + 1}"
    return rec

def _normalize_generic_record(item: Any, idx: int, fallback_prefix: str) -> Dict[str, Any]:
    if isinstance(item, dict):
        rec = dict(item)
    else:
        rec = {"value": str(item)}

    rec["title"] = (
        rec.get("title")
        or rec.get("content")
        or rec.get("summary")
        or rec.get("value")
        or rec.get("id")
        or f"{fallback_prefix} {idx + 1}"
    )
    return rec

def _episodic_snapshot(pm) -> Dict[str, Any]:
    result = {
        "summary": {},
        "recent_turns": [],
        "search_results": [],
    }

    if pm is None:
        return result

    try:
        stats = pm.get_stats()
        if isinstance(stats, dict):
            working_stats = stats.get("working", {}) or {}
            episodic_stats = stats.get("episodic", {}) or {}

            if "message_count" in working_stats:
                result["summary"]["Working messages"] = working_stats["message_count"]
            if "total_tokens" in working_stats:
                result["summary"]["Working tokens"] = working_stats["total_tokens"]
            if "episode_count" in episodic_stats:
                result["summary"]["Episodes"] = episodic_stats["episode_count"]
            if "enabled" in episodic_stats:
                result["summary"]["Episodic enabled"] = episodic_stats["enabled"]
    except Exception:
        pass

    try:
        recent_turns = pm.get_recent_turns(n=12)
        result["recent_turns"] = [
            _normalize_message(item, idx)
            for idx, item in enumerate(recent_turns or [])
        ]
    except Exception:
        pass

    return result


def _semantic_snapshot(pm) -> Dict[str, Any]:
    result = {
        "summary": {},
        "facts": [],
        "preferences": [],
        "events": [],
        "available_methods": [],
        "graph_stats": {},
    }

    if pm is None:
        return result

    semantic = getattr(pm, "semantic", None)
    if semantic is None:
        return result

    result["available_methods"] = [m for m in dir(semantic) if not m.startswith("_")][:200]

    try:
        graph_stats = semantic.get_stats()
        if isinstance(graph_stats, dict):
            result["graph_stats"] = graph_stats
            node_counts = graph_stats.get("node_counts", {}) or {}
            rel_counts = graph_stats.get("rel_counts", {}) or {}
            result["summary"]["Node tables"] = len(graph_stats.get("node_tables", []) or [])
            result["summary"]["Relationship tables"] = len(graph_stats.get("rel_tables", []) or [])
            result["summary"]["Total nodes"] = sum(int(v) for v in node_counts.values()) if node_counts else 0
            result["summary"]["Total relationships"] = sum(int(v) for v in rel_counts.values()) if rel_counts else 0
    except Exception:
        pass

    for label, method_name, out_key in (
        ("Facts", "list_facts", "facts"),
        ("Preferences", "list_preferences", "preferences"),
        ("Events", "list_events", "events"),
    ):
        if not hasattr(semantic, method_name):
            continue

        method = getattr(semantic, method_name)
        try:
            items = method(limit=50)
        except TypeError:
            try:
                items = method()
            except Exception:
                items = []
        except Exception:
            items = []

        items = items or []
        result["summary"][label] = len(items)
        result[out_key] = [
            _normalize_generic_record(item, idx, label[:-1])
            for idx, item in enumerate(items)
        ]

    return result

def _neural_snapshot(pm) -> Dict[str, Any]:
    result = {
        "summary": {},
        "stats": {},
        "available_methods": [],
        "enabled": False,
    }

    if pm is None:
        return result

    neural = getattr(pm, "neural", None)
    if neural is None:
        return result

    result["enabled"] = True
    result["available_methods"] = [m for m in dir(neural) if not m.startswith("_")][:200]

    try:
        stats = neural.get_stats()
        if isinstance(stats, dict):
            result["stats"] = stats
            for key, label in (
                ("embedding_dim", "Embedding dim"),
                ("hidden_dim", "Hidden dim"),
                ("memory_size", "Memory size"),
                ("total_updates", "Total updates"),
                ("trained", "Trained"),
            ):
                if key in stats:
                    result["summary"][label] = stats[key]
    except Exception:
        pass

    return result

def _graph_snapshot(pm) -> Dict[str, Any]:
    result = {
        "summary": {},
        "stats": {},
        "rows": [],
        "extractor": {},
    }

    if pm is None:
        return result

    semantic = getattr(pm, "semantic", None)
    if semantic is None:
        return result

    try:
        stats = semantic.get_stats()
        if isinstance(stats, dict):
            result["stats"] = stats
            node_counts = stats.get("node_counts", {}) or {}
            rel_counts = stats.get("rel_counts", {}) or {}

            result["summary"]["Node tables"] = len(stats.get("node_tables", []) or [])
            result["summary"]["Relationship tables"] = len(stats.get("rel_tables", []) or [])
            result["summary"]["Total nodes"] = sum(int(v) for v in node_counts.values()) if node_counts else 0
            result["summary"]["Total relationships"] = sum(int(v) for v in rel_counts.values()) if rel_counts else 0

            rows = []
            for name, count in node_counts.items():
                rows.append({"kind": "node_table", "name": name, "count": count, "title": f"Node table: {name}"})
            for name, count in rel_counts.items():
                rows.append({"kind": "relationship_table", "name": name, "count": count, "title": f"Relationship table: {name}"})
            result["rows"] = rows
    except Exception:
        pass

    # Extractor stats (populated by index_text / index_documents)
    extractor = getattr(pm, "extractor", None)
    if extractor is not None:
        try:
            result["extractor"] = extractor.get_stats()
            ex = result["extractor"]
            result["summary"]["Extraction entities"] = ex.get("entities", 0)
            result["summary"]["Extraction edges"] = ex.get("co_occurs_edges", 0)
            result["summary"]["Extraction method"] = ex.get("method", "")
        except Exception:
            pass

    return result

def _retrieval_trace_snapshot(pm, session_state) -> Dict[str, Any]:
    result = {
        "summary": {},
        "query": None,
        "prompt_preview": None,
        "context": None,
        "trace": None,
        "diagnostics": None,
    }

    result["query"] = session_state.get("last_user_query")
    result["prompt_preview"] = session_state.get("last_prompt_preview")
    result["context"] = session_state.get("last_context_dict")
    result["trace"] = session_state.get("last_retrieval_trace")

    if isinstance(result["context"], dict):
        token_counts = result["context"].get("token_counts", {}) or {}
        for key, label in (
            ("working", "Working tokens"),
            ("episodic", "Episodic tokens"),
            ("semantic", "Semantic tokens"),
            ("cold", "Cold tokens"),
            ("total", "Total context tokens"),
        ):
            if key in token_counts:
                result["summary"][label] = token_counts[key]

        result["summary"]["Working items"] = len(result["context"].get("working", []) or [])
        result["summary"]["Episodic items"] = len(result["context"].get("episodic", []) or [])
        result["summary"]["Semantic items"] = len(result["context"].get("semantic", []) or [])
        result["summary"]["Cold items"] = len(result["context"].get("cold", []) or [])

    if isinstance(result["trace"], dict):
        for key, label in (
            ("prompt_tokens", "Prompt tokens"),
            ("memory_tokens", "Memory tokens"),
            ("compressed", "Compressed"),
        ):
            if key in result["trace"]:
                result["summary"][label] = result["trace"][key]

    if pm is not None:
        try:
            result["diagnostics"] = pm.get_diagnostics_snapshot()
        except Exception:
            pass

    return result
    
def _context_snapshot(session_state) -> Dict[str, Any]:
    context = session_state.get("last_context_dict") or {}
    trace = session_state.get("last_retrieval_trace") or {}

    result = {
        "query": session_state.get("last_user_query"),
        "prompt_preview": session_state.get("last_prompt_preview"),
        "context": context,
        "prompt_sections": {},
        "summary": {},
    }

    if isinstance(trace, dict):
        prompt_sections = trace.get("prompt_sections")
        if isinstance(prompt_sections, dict):
            result["prompt_sections"] = prompt_sections

    if isinstance(context, dict):
        token_counts = context.get("token_counts", {}) or {}
        for key, label in (
            ("working", "Working tokens"),
            ("episodic", "Episodic tokens"),
            ("semantic", "Semantic tokens"),
            ("cold", "Cold tokens"),
            ("total", "Total context tokens"),
        ):
            if key in token_counts:
                result["summary"][label] = token_counts[key]

        for key, label in (
            ("working", "Working items"),
            ("episodic", "Episodic items"),
            ("semantic", "Semantic items"),
            ("cold", "Cold items"),
        ):
            result["summary"][label] = len(context.get(key, []) or [])

    return result


def _normalize_context_item(item: Any, idx: int, fallback_prefix: str) -> Dict[str, Any]:
    if isinstance(item, dict):
        rec = dict(item)
    else:
        rec = {"value": str(item)}

    rec["title"] = (
        rec.get("title")
        or rec.get("summary")
        or rec.get("content")
        or rec.get("text")
        or rec.get("id")
        or f"{fallback_prefix} {idx + 1}"
    )
    return rec


def _render_context_tier(title: str, items: List[Any], empty_text: str) -> None:
    normalized = [
        _normalize_context_item(item, idx, title[:-1] if title.endswith("s") else title)
        for idx, item in enumerate(items or [])
    ]
    _render_record_list(normalized, title, empty_text)


def _render_prompt_sections(prompt_sections: Dict[str, Any]) -> None:
    st.markdown("**Prompt sections**")
    if not prompt_sections:
        st.caption("No prompt sections captured.")
        return

    for key, value in prompt_sections.items():
        with st.expander(str(key), expanded=False):
            if isinstance(value, str):
                st.code(value, language="text")
            else:
                st.json(value)


def _render_episodic_tab(pm) -> None:
    st.caption("Working and episodic memory show the recent conversation plus semantically retrieved past episodes.")
    snap = _episodic_snapshot(pm)
    _render_summary_metrics("Episodic summary", snap.get("summary", {}))

    search_query = st.text_input("Search episodic memory", key="memory_inspector_episode_query")
    search_n = st.slider("Episode search results", min_value=1, max_value=20, value=8, key="memory_inspector_episode_limit")

    if search_query and pm is not None:
        try:
            hits = pm.search_episodes(search_query, n=search_n)
            snap["search_results"] = [
                _normalize_episode(item, idx)
                for idx, item in enumerate(hits or [])
            ]
        except Exception as exc:
            st.warning(f"Episode search unavailable: {exc}")

    _render_record_list(
        snap.get("recent_turns", []),
        "Recent working-memory turns",
        "No recent turns available.",
    )

    if search_query:
        _render_record_list(
            snap.get("search_results", []),
            "Episodic search results",
            "No episodic search results.",
        )


def _render_semantic_tab(pm) -> None:
    st.caption("Semantic memory captures reusable facts, preferences, and events that can persist beyond one conversation.")
    snap = _semantic_snapshot(pm)
    _render_summary_metrics("Semantic summary", snap.get("summary", {}))

    facts_col, prefs_col, events_col = st.columns(3)

    with facts_col:
        _render_record_list(
            snap.get("facts", []),
            "Facts",
            "No facts available.",
        )

    with prefs_col:
        _render_record_list(
            snap.get("preferences", []),
            "Preferences",
            "No preferences available.",
        )

    with events_col:
        _render_record_list(
            snap.get("events", []),
            "Events",
            "No events available.",
        )

    if snap.get("graph_stats"):
        with st.expander("Semantic graph stats", expanded=False):
            st.json(snap["graph_stats"])

    if not snap.get("facts") and not snap.get("preferences") and not snap.get("events"):
        with st.expander("Semantic debug", expanded=False):
            st.write("Available semantic methods:", snap.get("available_methods", []))



def _render_retrieval_trace_tab(pm, session_state) -> None:
    st.caption("Retrieval Trace explains what context was assembled for the last prompt and how much of each memory tier contributed.")
    snap = _retrieval_trace_snapshot(pm, session_state)
    _render_summary_metrics("Retrieval trace summary", snap.get("summary", {}))

    if snap.get("query"):
        st.markdown("**Last user query**")
        st.code(str(snap["query"]), language="text")

    if snap.get("prompt_preview"):
        with st.expander("Prompt preview", expanded=False):
            st.code(str(snap["prompt_preview"]), language="text")

    st.markdown("**Assembled context**")
    context = snap.get("context")
    if context is None:
        st.caption("No assembled context recorded yet.")
    else:
        st.json(context)

    st.markdown("**Last retrieval trace**")
    trace = snap.get("trace")
    if trace is None:
        st.caption("No retrieval trace recorded yet.")
    elif isinstance(trace, dict):
        st.json(trace)
    else:
        st.code(str(trace), language="text")

    st.markdown("**Diagnostics snapshot**")
    diagnostics = snap.get("diagnostics")
    if diagnostics is None:
        st.caption("No diagnostics snapshot available.")
    elif isinstance(diagnostics, dict):
        st.json(diagnostics)
    else:
        st.code(str(diagnostics), language="text")
        
def _render_graph_tab(pm) -> None:
    st.caption("The graph view shows the semantic layer as a property-graph structure: node types, relationship types, and counts.")
    snap = _graph_snapshot(pm)
    _render_summary_metrics("Graph summary", snap.get("summary", {}))
    _render_record_list(
        snap.get("rows", []),
        "Graph structure",
        "No graph structure data available.",
    )

    # Extractor panel
    extractor = getattr(pm, "extractor", None)
    if extractor is not None:
        st.markdown("**Graph extraction** (zero-cost TF-IDF → Kuzu)")
        ex_stats = snap.get("extractor", {})
        if ex_stats:
            col1, col2, col3 = st.columns(3)
            col1.metric("Entities", ex_stats.get("entities", 0))
            col2.metric("Co-occur edges", ex_stats.get("co_occurs_edges", 0))
            col3.metric("Method", ex_stats.get("method", "tfidf"))

        with st.expander("Index text into graph", expanded=False):
            text_input = st.text_area(
                "Text to index",
                height=120,
                placeholder="Paste document text here to extract entities and relations into the semantic graph.",
                key="graph_index_text_input",
            )
            method = st.selectbox("Extraction method", ["tfidf", "spacy"], key="graph_index_method")
            if st.button("Index text", key="graph_index_btn"):
                if text_input.strip():
                    with st.spinner("Extracting…"):
                        try:
                            from engram.memory.extraction import ExtractionConfig
                            cfg = ExtractionConfig(method=method)
                            stats = pm.index_text(text_input, config=cfg)
                            st.success(
                                f"Indexed: {stats.entities} entities, "
                                f"{stats.relations} relations, "
                                f"{stats.sentences} sentences "
                                f"({stats.elapsed_s:.2f}s)"
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                else:
                    st.warning("Enter some text first.")
    else:
        st.caption("Graph extraction requires semantic memory (pip install engram[semantic]).")

    if snap.get("stats"):
        with st.expander("Raw graph stats", expanded=False):
            st.json(snap["stats"])

def _render_neural_tab(pm) -> None:
    st.caption("Neural memory surfaces learned or latent state that complements explicit symbolic memory.")
    snap = _neural_snapshot(pm)

    if not snap.get("enabled"):
        st.caption("Neural memory is not enabled for this project.")
        return

    _render_summary_metrics("Neural summary", snap.get("summary", {}))

    if snap.get("stats"):
        st.json(snap["stats"])

    with st.expander("Neural debug", expanded=False):
        st.write("Available neural methods:", snap.get("available_methods", []))

   
def _render_context_tab(session_state) -> None:
    st.caption("Assembled Context shows the actual memory content selected for the last prompt, grouped by memory tier.")
    snap = _context_snapshot(session_state)
    _render_summary_metrics("Context summary", snap.get("summary", {}))

    if snap.get("query"):
        st.markdown("**Last user query**")
        st.code(str(snap["query"]), language="text")

    context = snap.get("context") or {}
    if not context:
        st.caption("No assembled context recorded yet.")
        return

    working_items = context.get("working", []) or []
    episodic_items = context.get("episodic", []) or []
    semantic_items = context.get("semantic", []) or []
    cold_items = context.get("cold", []) or []

    working_tab, episodic_tab, semantic_tab, cold_tab = st.tabs(
        ["Working", "Episodic", "Semantic", "Cold"]
    )

    with working_tab:
        _render_context_tier(
            "Working items",
            working_items,
            "No working-memory items in the assembled context.",
        )

    with episodic_tab:
        _render_context_tier(
            "Episodic items",
            episodic_items,
            "No episodic items in the assembled context.",
        )

    with semantic_tab:
        _render_context_tier(
            "Semantic items",
            semantic_items,
            "No semantic items in the assembled context.",
        )

    with cold_tab:
        _render_context_tier(
            "Cold items",
            cold_items,
            "No cold-storage items in the assembled context.",
        )

    _render_prompt_sections(snap.get("prompt_sections", {}))

    if snap.get("prompt_preview"):
        with st.expander("Prompt preview", expanded=False):
            st.code(str(snap["prompt_preview"]), language="text")

    with st.expander("Raw context dump", expanded=False):
        st.json(context)    


def _maintenance_snapshot(pm) -> Dict[str, Any]:
    result: Dict[str, Any] = {"forgetting": {}, "embedding_cache": {}}
    if pm is None:
        return result
    forgetting = getattr(pm, "forgetting", None)
    if forgetting is not None:
        try:
            result["forgetting"] = forgetting.get_stats()
        except Exception:
            pass
    emb_cache = getattr(pm, "embedding_cache", None)
    if emb_cache is not None:
        try:
            result["embedding_cache"] = emb_cache.get_stats()
        except Exception:
            pass
    return result


def _render_maintenance_tab(pm) -> None:
    st.caption("Memory maintenance controls: forgetting policy and embedding cache.")

    snap = _maintenance_snapshot(pm)

    # --- Forgetting policy ---
    st.markdown("**Forgetting policy** (episodic → cold storage)")
    fg = snap.get("forgetting", {})
    if fg:
        col1, col2, col3 = st.columns(3)
        col1.metric("Enabled", "yes" if fg.get("enabled") else "no")
        col2.metric("Retention threshold", fg.get("retention_threshold", "—"))
        col3.metric("Episodes since last run", fg.get("episodes_since_last_run", 0))

        tracker = fg.get("access_tracker", {})
        if tracker:
            col4, col5 = st.columns(2)
            col4.metric("Tracked episodes", tracker.get("tracked_episodes", 0))
            col5.metric("Total accesses", tracker.get("total_accesses", 0))

        col_dry, col_run = st.columns(2)
        with col_dry:
            if st.button("Dry run", key="maintenance_dry_run"):
                with st.spinner("Scoring episodes…"):
                    try:
                        result = pm.run_maintenance(dry_run=True)
                        st.json(result)
                    except Exception as exc:
                        st.error(str(exc))
        with col_run:
            if st.button("Run now", type="primary", key="maintenance_run"):
                with st.spinner("Archiving low-retention episodes…"):
                    try:
                        result = pm.run_maintenance(dry_run=False)
                        archived = result.get("archived", 0)
                        if archived:
                            st.success(f"Archived {archived} episodes to cold storage.")
                        else:
                            st.info(f"Nothing to archive. {result.get('reason', result.get('status', ''))}")
                        st.json(result)
                    except Exception as exc:
                        st.error(str(exc))
    else:
        st.caption("Forgetting policy not available.")

    st.divider()

    # --- Embedding cache ---
    st.markdown("**Embedding cache**")
    ec = snap.get("embedding_cache", {})
    if ec:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Enabled", "yes" if ec.get("enabled") else "no")
        col2.metric("Memory entries", ec.get("memory_size", 0))
        col3.metric("Hit rate", f"{ec.get('hit_rate', 0.0):.1%}")
        col4.metric("Disk cache", "yes" if ec.get("has_disk") else "no")

        if ec.get("enabled"):
            if st.button("Clear cache", key="cache_clear_btn"):
                try:
                    pm.embedding_cache.clear()
                    st.success("Embedding cache cleared.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        with st.expander("Raw cache stats", expanded=False):
            st.json(ec)
    else:
        st.caption("Embedding cache not available.")



def render_memory_tab(pm, session_state) -> None:
    st.header("Memory")
    st.caption("This tab is both a debugging console and a reference implementation for how layered memory can be inspected in an application.")

    config_snapshot = _runtime_memory_config_snapshot(pm)
    _render_summary_metrics("Memory runtime configuration", config_snapshot)

    episodic_tab, semantic_tab, graph_tab, neural_tab, context_tab, trace_tab, maintenance_tab = st.tabs(
        ["Episodic", "Semantic", "Graph", "Neural", "Assembled Context", "Retrieval Trace", "Maintenance"]
    )

    with episodic_tab:
        _render_episodic_tab(pm)

    with semantic_tab:
        _render_semantic_tab(pm)

    with graph_tab:
        _render_graph_tab(pm)

    with neural_tab:
        _render_neural_tab(pm)

    with context_tab:
        _render_context_tab(session_state)

    with trace_tab:
        _render_retrieval_trace_tab(pm, session_state)

    with maintenance_tab:
        _render_maintenance_tab(pm)
