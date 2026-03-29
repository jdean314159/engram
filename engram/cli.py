"""
Engram command-line interface.

Commands:
  engram doctor     Diagnose environment and engine/profile configuration.
  engram recommend  Recommend a model/profile based on best-effort hardware detection.

This CLI is intentionally dependency-light (uses stdlib + PyYAML only).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


DEFAULT_CONFIG_REL = Path(__file__).parent / "engine" / "llm_engines.yaml"


def _print_kv(key: str, value: str) -> None:
    print(f"{key:<22} {value}")


def _http_json(
    method: str,
    url: str,
    payload: Optional[dict] = None,
    timeout_s: int = 10,
) -> Tuple[int, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            if not raw:
                return resp.status, None
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except Exception:
                return resp.status, raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return e.code, body
    except Exception as e:
        return 0, str(e)


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _sqlite_has_fts5() -> bool:
    import sqlite3
    try:
        con = sqlite3.connect(":memory:")
        cur = con.cursor()
        cur.execute("CREATE VIRTUAL TABLE t USING fts5(content);")
        con.close()
        return True
    except Exception:
        return False


@dataclass
class HardwareInfo:
    ram_gb: Optional[float]
    accel: str  # "cuda", "mps", "cpu"
    vram_gb: Optional[float]
    free_vram_gb: Optional[float] = None
    
def _get_nvidia_smi_memory_best_effort() -> Tuple[Optional[float], Optional[float]]:
    try:
        import subprocess
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()

        if not out:
            return None, None

        first_line = out.splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 2:
            return None, None

        total_gb = float(parts[0]) / 1024.0
        free_gb = float(parts[1]) / 1024.0
        return total_gb, free_gb
    except Exception:
        return None, None    


def _get_ram_gb_best_effort() -> Optional[float]:
    # Cross-platform best-effort: prefer psutil if available, otherwise platform-specific fallbacks.
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024 ** 3)
        except Exception:
            return None

    if sys.platform == "darwin":
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out) / (1024 ** 3)
        except Exception:
            return None

    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            return None

    return None


def _detect_hardware() -> HardwareInfo:
    ram = _get_ram_gb_best_effort()

    accel = "cpu"
    vram = None
    free_vram = None

    smi_total, smi_free = _get_nvidia_smi_memory_best_effort()

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            accel = "cuda"
            try:
                props = torch.cuda.get_device_properties(0)
                vram = float(props.total_memory) / (1024 ** 3)
            except Exception:
                vram = smi_total

            free_vram = smi_free

        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            accel = "mps"
            if ram is not None:
                vram = max(0.0, min(ram * 0.75, ram - 4.0))
            else:
                vram = None
            free_vram = None

        elif smi_total is not None:
            # GPU exists, but PyTorch could not use it; keep CLI behavior conservative.
            accel = "cpu"
            vram = smi_total
            free_vram = smi_free

    except Exception:
        if smi_total is not None:
            vram = smi_total
            free_vram = smi_free

    return HardwareInfo(ram_gb=ram, accel=accel, vram_gb=vram, free_vram_gb=free_vram)


def _engine_url(engine_cfg: dict) -> Optional[str]:
    base = engine_cfg.get("base_url")
    if not base:
        return None
    # Many configs use /v1; Ollama's version endpoint is on /api/version.
    # We'll strip a trailing /v1 if present.
    if base.endswith("/v1"):
        return base[:-3]
    return base.rstrip("/")


def _ollama_version(base_url: str) -> Tuple[bool, str]:
    # Returns (reachable, version_str)
    status, body = _http_json("GET", f"{base_url}/api/version", timeout_s=5)
    if status == 200 and isinstance(body, dict) and "version" in body:
        return True, str(body["version"])
    if status == 0:
        return False, str(body)
    return True, "unknown"


def _ollama_model_present(base_url: str, model: str) -> Tuple[bool, str]:
    status, body = _http_json("POST", f"{base_url}/api/show", payload={"name": model}, timeout_s=10)
    if status == 200:
        return True, "present"
    if status in (404, 400):
        return False, "missing"
    return False, f"error: {status} {body}"


def _ollama_pull_model(base_url: str, model: str, *, verbose: bool = True) -> bool:
    # /api/pull streams JSON lines; we will read until done or error.
    try:
        payload = json.dumps({"name": model}).encode("utf-8")
        req = urllib.request.Request(
            url=f"{base_url}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            # /api/pull streams JSON objects. Update a single progress line when totals are present,
            # otherwise print only on status changes.
            done = False
            last_msg: Optional[dict] = None
            last_status: Optional[str] = None
            last_print_t = 0.0

            def _print_progress(status: str, completed: Optional[int], total: Optional[int]) -> None:
                nonlocal last_print_t
                if not verbose:
                    return
                now = time.time()
                # Rate limit to avoid terminal spam.
                if now - last_print_t < 0.15:
                    return
                last_print_t = now

                if completed is not None and total:
                    pct = max(0.0, min(100.0, (float(completed) / float(total)) * 100.0))
                    # Carriage return keeps it to a single line.
                    if verbose: print(f"\r  pull: {pct:5.1f}%  {status}", end="", flush=True)
                else:
                    if verbose: print(f"\r  pull: {status}                          ", end="", flush=True)

            for raw_line in resp:
                try:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        last_msg = obj
                    status = str(obj.get("status")) if isinstance(obj, dict) and obj.get("status") else None
                    completed = obj.get("completed") if isinstance(obj, dict) else None
                    total = obj.get("total") if isinstance(obj, dict) else None

                    if status:
                        # Only print when status changes, or when we have progress numbers.
                        if (completed is not None and total is not None) or status != last_status:
                            _print_progress(status, completed, total)
                            last_status = status

                        if status == "success":
                            done = True
                            break
                except Exception:
                    continue

            # Ensure we end the progress line.
            if verbose: print("", flush=True)
            return done or (isinstance(last_msg, dict) and last_msg.get("status") == "success")
    except Exception as e:
        print(f"  pull error: {e}")
        return False


def cmd_doctor(args: argparse.Namespace) -> int:
    verbose = getattr(args, "verbose", False)
    config_path = Path(args.config).expanduser().resolve()
    cfg = _load_config(config_path)

    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "sqlite_fts5": _sqlite_has_fts5(),
    }

    hw = _detect_hardware()
    report["hardware"] = {
        "ram_gb": hw.ram_gb,
        "accelerator": hw.accel,
        "vram_gb": hw.vram_gb,
        "free_vram_gb": hw.free_vram_gb,
    }

    if not args.json:
        _print_kv("python", report["python"])
        _print_kv(
            "platform",
            f"{report['platform']['system']} {report['platform']['release']} ({report['platform']['machine']})",
        )
        _print_kv("sqlite_fts5", "yes" if report["sqlite_fts5"] else "no (fallback search will be used)")

        ram_s = f"{hw.ram_gb:.1f} GB" if hw.ram_gb is not None else "unknown"
        vram_s = f"{hw.vram_gb:.1f} GB" if hw.vram_gb is not None else ("n/a" if hw.accel != "cuda" else "unknown")
        _print_kv("ram", ram_s)
        _print_kv("accelerator", hw.accel)
        _print_kv("vram", vram_s)

    profiles = (cfg.get("profiles") or {})
    engines = (cfg.get("engines") or {})

    profile_name = args.profile
    if profile_name not in profiles:
        report["error"] = {
            "type": "profile_not_found",
            "profile": profile_name,
            "available_profiles": sorted(profiles.keys()),
        }
        if args.json:
            if verbose: print(json.dumps(report, indent=2, sort_keys=True))
        else:
            if verbose: print()
            if verbose: print(f"Profile not found: {profile_name}")
            if verbose: print("Available profiles:")
            for p in sorted(profiles.keys()):
                print(f"  - {p}")
        return 2

    prof = profiles[profile_name] or {}
    prof_engines = prof.get("engines") or []
    report["profile"] = {
        "name": profile_name,
        "engine_count": len(prof_engines),
        "engines": [],
    }

    if not args.json:
        print()
        _print_kv("profile", profile_name)
        _print_kv("engine_count", str(len(prof_engines)))

    for key in prof_engines:
        e = engines.get(key)
        if not e:
            report["profile"]["engines"].append({"name": key, "error": "missing_config"})
            if not args.json:
                print(f"- engine={key}: MISSING CONFIG")
            continue
        etype = str(e.get("type", "unknown"))
        model = str(e.get("model", ""))
        base = _engine_url(e)

        engine_report: Dict[str, Any] = {
            "name": key,
            "type": etype,
            "model": model or None,
            "base_url": base,
        }

        if not args.json:
            line = f"- engine={key} type={etype}"
            if model:
                line += f" model={model}"
            if verbose: print(line)

        if etype == "ollama" and base:
            ok, ver = _ollama_version(base)
            engine_report["ollama"] = {"reachable": ok, "version": ver}
            if not args.json:
                _print_kv("  ollama_reachable", "yes" if ok else "no")
                _print_kv("  ollama_version", ver)

            present, msg = _ollama_model_present(base, model)
            engine_report["model_status"] = msg
            if not args.json:
                _print_kv("  model", msg)

            if args.pull_missing and ok and not present:
                if not args.json:
                    if verbose: print("  action              pulling model (this may take a while)...")
                pull_ok = _ollama_pull_model(base, model)
                engine_report["pull_result"] = "success" if pull_ok else "failed"
                if not args.json:
                    _print_kv("  pull_result", "success" if pull_ok else "failed")

        elif etype in ("openai", "anthropic", "vllm"):
            if base:
                if not args.json:
                    _print_kv("  base_url", base)
            if etype == "openai":
                env = e.get("api_key_env") or "ENGRAM_OPENAI_API_KEY"
                key_set = bool(os.environ.get(str(env), ""))
                engine_report["api_key_env"] = str(env)
                engine_report["api_key_set"] = key_set
                if not args.json:
                    _print_kv("  api_key_env", f"{env} ({'set' if key_set else 'not set'})")

        report["profile"]["engines"].append(engine_report)

    if args.recommend:
        report["recommendation"] = _recommend_from_hw(hw, cfg, profile_name)
        if not args.json:
            if verbose: print()
            _print_kv("recommendation", report["recommendation"])

    # Optional: attach a telemetry summary (best-effort). This is useful for issue reports.
    telemetry_summary = _summarize_telemetry_best_effort(cfg, profile_name)
    if telemetry_summary is not None:
        report["telemetry_summary"] = telemetry_summary

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.jsonl:
        out_path = Path(args.jsonl).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.json:
            if verbose: print()
            _print_kv("wrote_report", str(out_path))
    return 0


def _recommend_from_hw(hw: HardwareInfo, cfg: dict, current_profile: str) -> str:
    """
    Best-effort recommendation: prints a human-readable suggestion without mutating config.

    Heuristic targets:
      - If CUDA VRAM >= 20 GB: primary 32B OK, fallback 8B/7B.
      - If CUDA VRAM 10-20 GB: primary 14B-ish, fallback 7B/8B.
      - If CUDA VRAM < 10 GB or CPU-only: use small local model or enable cloud.
    """
    profiles = (cfg.get("profiles") or {})
    engines = (cfg.get("engines") or {})

    def profile_models(pname: str) -> str:
        p = profiles.get(pname) or {}
        names = p.get("engines") or []
        models = []
        for n in names:
            e = engines.get(n) or {}
            m = e.get("model")
            if m:
                models.append(str(m))
        return ", ".join(models) if models else "(no models)"

    current_models = profile_models(current_profile)

    if hw.accel in ("cuda", "mps") and hw.vram_gb is not None:
        accel_label = "GPU" if hw.accel == "cuda" else "Apple Silicon unified memory"

        occupied_note = ""
        if hw.accel == "cuda" and hw.free_vram_gb is not None and hw.free_vram_gb < 2.0:
            occupied_note = (
                f" Current free GPU memory is very low (~{hw.free_vram_gb:.1f} GB), "
                "which suggests another model/runtime is already loaded."
            )
        elif hw.accel == "cuda" and hw.free_vram_gb is not None and hw.free_vram_gb < 8.0:
            occupied_note = (
                f" Current free GPU memory is limited (~{hw.free_vram_gb:.1f} GB), "
                "so switching models may require stopping the active runtime first."
            )

        if hw.vram_gb >= 20:
            return (
                f"Your {accel_label} budget (~{hw.vram_gb:.1f} GB) should handle a large primary model in principle."
                f"{occupied_note} "
                f"Current profile '{current_profile}' uses: {current_models}. "
                "If you see OOM, reduce primary size or lower context/max_output."
            )

        if 10 <= hw.vram_gb < 20:
            return (
                f"Your {accel_label} budget (~{hw.vram_gb:.1f} GB) is mid-range."
                f"{occupied_note} "
                "Consider a medium primary (≈14B) with a 7B/8B fallback. "
                f"Current profile '{current_profile}' uses: {current_models}."
            )

        return (
            f"Your {accel_label} budget (~{hw.vram_gb:.1f} GB) is likely tight for very large models."
            f"{occupied_note} "
            "Prefer a 7B/8B primary and keep prompts short, or enable cloud failover for hard queries. "
            f"Current profile '{current_profile}' uses: {current_models}."
        )

    if hw.accel == "mps":
        return (
            "Apple Silicon (MPS) detected. Prefer a smaller local model (7B/8B) and conservative context sizes. "
            f"Current profile '{current_profile}' uses: {current_models}."
        )

    return (
        "CPU-only runtime detected (or GPU not visible to PyTorch). Prefer a small local model and strict prompt budgets, "
        "or enable cloud failover if you need larger context/quality. "
        f"Current profile '{current_profile}' uses: {current_models}."
    )


def _suggest_alternatives(model: str) -> list[str]:
    m = (model or "").lower()
    # Heuristic suggestions only; we don't try to be prescriptive.
    if "32" in m and "b" in m:
        return [m.replace("32", "14"), m.replace("32", "8"), m.replace("32", "7")]
    if "14" in m and "b" in m:
        return [m.replace("14", "8"), m.replace("14", "7")]
    if "13" in m and "b" in m:
        return [m.replace("13", "7"), m.replace("13", "8")]
    return []


def _check_profile_models_no_pull(cfg: dict, profile: str) -> dict:
    profiles = (cfg.get("profiles") or {})
    engines = (cfg.get("engines") or {})
    prof = profiles.get(profile) or {}
    out = {"profile": profile, "checks": []}
    for key in (prof.get("engines") or []):
        e = engines.get(key) or {}
        etype = str(e.get("type", ""))
        model = str(e.get("model", ""))
        base = _engine_url(e)
        if etype == "ollama" and base and model:
            ok, _ver = _ollama_version(base)
            present, msg = _ollama_model_present(base, model)
            out["checks"].append(
                {
                    "engine": key,
                    "type": "ollama",
                    "base_url": base,
                    "reachable": ok,
                    "model": model,
                    "status": msg,
                    "suggested_alternatives": _suggest_alternatives(model) if (ok and not present) else [],
                }
            )
    return out


def cmd_recommend(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    cfg = _load_config(config_path)
    hw = _detect_hardware()
    profile = args.profile
    rec = _recommend_from_hw(hw, cfg, profile)
    model_checks = _check_profile_models_no_pull(cfg, profile)
    payload = {"profile": profile, "recommendation": rec, "model_checks": model_checks}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(rec)
        # Also show model availability without downloading.
        checks = model_checks.get("checks") or []
        if checks:
            if verbose: print()
            if verbose: print("Model availability (no download):")
            for c in checks:
                status = c.get("status")
                reachable = c.get("reachable")
                eng = c.get("engine")
                model = c.get("model")
                base = c.get("base_url")
                print(f"- {eng}: {model} @ {base} -> {'unreachable' if not reachable else status}")
                alts = c.get("suggested_alternatives") or []
                if reachable and status == "missing" and alts:
                    if verbose: print(f"  suggested alternatives: {', '.join(alts[:3])}")

    if args.jsonl:
        out_path = Path(args.jsonl).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.json:
            if verbose: print()
            _print_kv("wrote_report", str(out_path))
    return 0


def _summarize_telemetry_best_effort(cfg: dict, profile: str) -> Optional[dict]:
    """Summarize recent telemetry events from a JSONL sink.

    Best-effort only: if no sink/path is configured, or the file can't be read, returns None.
    """
    # Priority: explicit env override, then profile telemetry config.
    env_path = os.environ.get("ENGRAM_TELEMETRY_PATH", "").strip()
    candidate: Optional[str] = env_path or None

    if not candidate:
        prof = (cfg.get("profiles") or {}).get(profile) or {}
        tel = prof.get("telemetry") or {}
        if bool(tel.get("enabled")) and str(tel.get("sink", "")).lower() == "jsonl":
            candidate = str(tel.get("jsonl_path") or "").strip() or None

    if not candidate:
        return None

    path = Path(candidate).expanduser()
    if not path.exists() or not path.is_file():
        return None

    # Read last ~1MB to keep it cheap.
    try:
        max_bytes = 1_000_000
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start)
            raw = f.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    lines = [ln for ln in raw.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"path": str(path), "events": 0}

    kinds: Dict[str, int] = {}
    last_ts: Optional[float] = None
    last_kind: Optional[str] = None

    # Parse from the end for "last" metadata.
    for ln in reversed(lines):
        try:
            obj = json.loads(ln)
            last_ts = float(obj.get("ts")) if obj.get("ts") is not None else None
            last_kind = str(obj.get("kind")) if obj.get("kind") is not None else None
            break
        except Exception:
            continue

    # Count kinds.
    for ln in lines:
        try:
            obj = json.loads(ln)
            k = str(obj.get("kind", "unknown"))
            kinds[k] = kinds.get(k, 0) + 1
        except Exception:
            kinds["parse_error"] = kinds.get("parse_error", 0) + 1

    return {
        "path": str(path),
        "events": len(lines),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
        "last": {"ts": last_ts, "kind": last_kind},
    }


def cmd_eval(args: argparse.Namespace) -> int:
    from engram.eval.harness import run_basic

    res = run_basic()
    if args.json:
        print(json.dumps(res, indent=2, sort_keys=True))
    else:
        ok = bool(res.get("ok"))
        print("Eval:", "PASS" if ok else "FAIL")
        for k, v in (res.get("checks") or {}).items():
            _print_kv(k, "ok" if v else "FAIL")
    return 0 if bool(res.get("ok")) else 2




# ---------------------------
# Programmatic API (used by UI)
# ---------------------------

def run_doctor(*, profile: str, pull_missing: bool = False, config: str = "~/.engram/llm_engines.yaml", include_recommend: bool = False) -> Dict[str, Any]:
    """Return the same dict payload as `engram doctor --json`.

    This is intended for programmatic use (e.g., Streamlit UI) so it MUST NOT print.
    """
    config_path = Path(config).expanduser().resolve()
    cfg = _load_config(config_path)

    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "sqlite_fts5": _sqlite_has_fts5(),
    }

    hw = _detect_hardware()
    report["hardware"] = {
        "ram_gb": hw.ram_gb,
        "accelerator": hw.accel,
        "vram_gb": hw.vram_gb,
        "free_vram_gb": hw.free_vram_gb,
    }

    profiles = (cfg.get("profiles") or {})
    engines = (cfg.get("engines") or {})

    if profile not in profiles:
        report["error"] = {
            "type": "profile_not_found",
            "profile": profile,
            "available_profiles": sorted(profiles.keys()),
        }
        return report

    prof = profiles[profile] or {}
    prof_engines = prof.get("engines") or []
    report["profile"] = {"name": profile, "engine_count": len(prof_engines), "engines": []}

    for key in prof_engines:
        e = engines.get(key)
        if not e:
            report["profile"]["engines"].append({"name": key, "error": "missing_config"})
            continue

        etype = str(e.get("type", "unknown"))
        model = str(e.get("model", "")) or None
        base = _engine_url(e)

        engine_report: Dict[str, Any] = {"name": key, "type": etype, "model": model, "base_url": base}

        if etype == "ollama" and base:
            ok, ver = _ollama_version(base)
            engine_report["ollama"] = {"reachable": ok, "version": ver}

            present, msg = _ollama_model_present(base, model or "")
            engine_report["model_present"] = present
            engine_report["model_status"] = msg

            if pull_missing and ok and (not present) and model:
                pulled = _ollama_pull_model(base, model, verbose=False)
                engine_report["pull_attempted"] = True
                engine_report["pull_success"] = bool(pulled)
        report["profile"]["engines"].append(engine_report)

    # Optional telemetry summary if configured (best-effort)
    report["telemetry_summary"] = _summarize_telemetry_best_effort(cfg, profile)

    if include_recommend:
        report["recommendation"] = run_recommend(profile=profile, config=str(config_path)).get("recommendation")
    return report


def run_recommend(*, profile: str, config: str = "~/.engram/llm_engines.yaml") -> Dict[str, Any]:
    """Return the same dict payload as `engram recommend --json`."""
    config_path = Path(config).expanduser().resolve()
    cfg = _load_config(config_path)

    hw = _detect_hardware()
    rec = _recommend_from_hw(hw, cfg, profile)

    profiles = (cfg.get("profiles") or {})
    engines = (cfg.get("engines") or {})

    model_checks: Dict[str, Any] = {}
    if profile in profiles:
        # NOTE: helper takes the full config dict + profile name.
        model_checks = _check_profile_models_no_pull(cfg, profile)

    return {
        "python": sys.version.split()[0],
        "hardware": {
            "ram_gb": hw.ram_gb,
            "accelerator": hw.accel,
            "vram_gb": hw.vram_gb,
            "free_vram_gb": hw.free_vram_gb,
        },
        "profile": profile,
        "recommendation": rec,
        "model_checks": model_checks,
    }

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="engram", description="Engram CLI")
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_REL),
        help="Path to llm_engines.yaml (default: bundled config)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="Diagnose engines/profile and environment")
    d.add_argument("--profile", default="default_local", help="Profile name from config")
    d.add_argument("--pull-missing", action="store_true", help="Pull missing Ollama models")
    d.add_argument("--recommend", action="store_true", help="Also print a profile/model recommendation")
    d.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    d.add_argument("--jsonl", help="Write the full doctor report JSON to a file path")
    d.set_defaults(func=cmd_doctor)

    r = sub.add_parser("recommend", help="Recommend a profile/model based on hardware")
    r.add_argument("--profile", default="default_local", help="Profile name to evaluate")
    r.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    r.add_argument("--jsonl", help="Write the recommendation JSON to a file path")
    r.set_defaults(func=cmd_recommend)

    e = sub.add_parser("eval", help="Run a lightweight CPU-only eval harness")
    e.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    e.set_defaults(func=cmd_eval)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
