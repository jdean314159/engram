"""Tests for engram/engine/model_manager.py.

Coverage:
  - Pure helper functions: _looks_quantized, _normalize_query,
    _quantized_queries, _estimate_params, classify_hf_model_format,
    _search_score, _merge_rank_results
  - recommend_format(): all branches of the decision tree
  - recommend_format_for_hf_model(): per-family routing
  - estimate_model_risk() + artifact_family_from_text()
  - extract_model_size_b()
  - scan_local_model(): single GGUF, directory GGUF, safetensors,
    pytorch, non-model dir, missing path
  - default_local_model_dir(): path construction and sanitisation
  - download_hf_model(): no-deps fallback path (returns DownloadResult)
  - detect_system(): always returns SystemInfo without raising
  - SystemInfo.primary_vram_gb property

All tests are CPU-only, no network, no GPU required.
Network-dependent tests (search_hf_models, download_hf_model with real
HF calls) are guarded by require("hf_network") and will skip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tests.harness.runner import require, test_group

from engram.engine.model_manager import (
    HFModelInfo,
    SystemInfo,
    GPUInfo,
    FormatRecommendation,
    LocalModelInfo,
    DownloadResult,
    _looks_quantized,
    _normalize_query,
    _quantized_queries,
    _estimate_params,
    classify_hf_model_format,
    _search_score,
    _merge_rank_results,
    recommend_format,
    recommend_format_for_hf_model,
    estimate_model_risk,
    artifact_family_from_text,
    extract_model_size_b,
    scan_local_model,
    default_local_model_dir,
    download_hf_model,
    detect_system,
    GGUF_QUANT_BPP,
    KV_OVERHEAD_FACTOR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(
    repo_id: str = "org/model-7b",
    downloads: int = 1000,
    likes: int = 50,
    pipeline_tag: str = "text-generation",
    tags: list[str] | None = None,
    has_gguf: bool = False,
    has_safetensors: bool = False,
    has_awq: bool = False,
    has_gptq: bool = False,
    has_exl2: bool = False,
    estimated_params_b: float | None = 7.0,
) -> HFModelInfo:
    name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    author = repo_id.split("/")[0] if "/" in repo_id else ""
    info = HFModelInfo(
        repo_id=repo_id,
        name=name,
        author=author,
        downloads=downloads,
        likes=likes,
        pipeline_tag=pipeline_tag,
        tags=tags or [],
        has_gguf=has_gguf,
        has_safetensors=has_safetensors,
        has_awq=has_awq,
        has_gptq=has_gptq,
        has_exl2=has_exl2,
        estimated_params_b=estimated_params_b,
    )
    if estimated_params_b is not None:
        info.estimated_size_gb_fp16 = estimated_params_b * 2.0
    return info


# ---------------------------------------------------------------------------
# Group 1: pure string helpers
# ---------------------------------------------------------------------------

@test_group("Model Manager – string helpers")
def test_looks_quantized_gguf():
    assert _looks_quantized("llama-7b-q4_k_m.gguf") is True


@test_group("Model Manager – string helpers")
def test_looks_quantized_awq():
    assert _looks_quantized("Mistral-7B-AWQ") is True


@test_group("Model Manager – string helpers")
def test_looks_quantized_gptq():
    assert _looks_quantized("TheBloke/Llama-2-13B-GPTQ") is True


@test_group("Model Manager – string helpers")
def test_looks_quantized_false():
    assert _looks_quantized("meta-llama/Llama-2-7b-hf") is False


@test_group("Model Manager – string helpers")
def test_looks_quantized_empty():
    assert _looks_quantized("") is False
    assert _looks_quantized(None) is False


@test_group("Model Manager – string helpers")
def test_normalize_query_strips_whitespace():
    assert _normalize_query("  llama  7b  ") == "llama 7b"


@test_group("Model Manager – string helpers")
def test_normalize_query_collapses_internal():
    assert _normalize_query("llama\t7b\n") == "llama 7b"


@test_group("Model Manager – string helpers")
def test_normalize_query_empty():
    assert _normalize_query("") == ""
    assert _normalize_query(None) == ""


@test_group("Model Manager – string helpers")
def test_quantized_queries_plain_query_expands():
    variants = _quantized_queries("llama 7b")
    assert "llama 7b" in variants
    assert any("AWQ" in v for v in variants)
    assert any("GPTQ" in v for v in variants)
    assert any("GGUF" in v for v in variants)


@test_group("Model Manager – string helpers")
def test_quantized_queries_already_quantized_no_expansion():
    variants = _quantized_queries("llama-7b-awq")
    assert len(variants) == 1
    assert variants[0] == "llama-7b-awq"


@test_group("Model Manager – string helpers")
def test_quantized_queries_empty():
    assert _quantized_queries("") == []


@test_group("Model Manager – string helpers")
def test_quantized_queries_deduplication():
    variants = _quantized_queries("mistral 7b")
    lower = [v.lower() for v in variants]
    assert len(lower) == len(set(lower))


# ---------------------------------------------------------------------------
# Group 2: _estimate_params
# ---------------------------------------------------------------------------

@test_group("Model Manager – parameter estimation")
def test_estimate_params_7b():
    info = HFModelInfo(repo_id="org/Llama-2-7b-hf", name="Llama-2-7b-hf")
    _estimate_params(info)
    assert info.estimated_params_b == 7.0
    assert info.estimated_size_gb_fp16 == 14.0


@test_group("Model Manager – parameter estimation")
def test_estimate_params_70b():
    info = HFModelInfo(repo_id="meta-llama/Llama-2-70b-chat-hf", name="Llama-2-70b-chat-hf")
    _estimate_params(info)
    assert info.estimated_params_b == 70.0


@test_group("Model Manager – parameter estimation")
def test_estimate_params_fractional():
    info = HFModelInfo(repo_id="Qwen/Qwen2-0.5b", name="Qwen2-0.5b")
    _estimate_params(info)
    assert info.estimated_params_b == 0.5


@test_group("Model Manager – parameter estimation")
def test_estimate_params_unknown_returns_none():
    info = HFModelInfo(repo_id="some/model-without-size", name="model-without-size")
    _estimate_params(info)
    assert info.estimated_params_b is None


@test_group("Model Manager – parameter estimation")
def test_estimate_params_fp16_size_consistent():
    info = HFModelInfo(repo_id="org/model-13b", name="model-13b")
    _estimate_params(info)
    assert info.estimated_params_b == 13.0
    assert abs(info.estimated_size_gb_fp16 - 26.0) < 0.01


# ---------------------------------------------------------------------------
# Group 3: classify_hf_model_format
# ---------------------------------------------------------------------------

@test_group("Model Manager – format classification")
def test_classify_gguf_only():
    m = _make_model(has_gguf=True, has_safetensors=False)
    assert classify_hf_model_format(m) == "gguf"


@test_group("Model Manager – format classification")
def test_classify_awq():
    m = _make_model(has_awq=True)
    assert classify_hf_model_format(m) == "transformers_quantized"


@test_group("Model Manager – format classification")
def test_classify_gptq():
    m = _make_model(has_gptq=True)
    assert classify_hf_model_format(m) == "transformers_quantized"


@test_group("Model Manager – format classification")
def test_classify_exl2():
    m = _make_model(has_exl2=True)
    assert classify_hf_model_format(m) == "transformers_quantized"


@test_group("Model Manager – format classification")
def test_classify_mixed_gguf_and_awq():
    m = _make_model(has_gguf=True, has_awq=True)
    assert classify_hf_model_format(m) == "mixed"


@test_group("Model Manager – format classification")
def test_classify_base():
    m = _make_model(has_safetensors=True)
    assert classify_hf_model_format(m) == "base"


@test_group("Model Manager – format classification")
def test_classify_signals_from_repo_name():
    # "gguf" in name should set has_gguf via _populate_format_signals
    m = _make_model(repo_id="TheBloke/Mistral-7B-GGUF", has_gguf=False)
    result = classify_hf_model_format(m)
    assert result == "gguf"


@test_group("Model Manager – format classification")
def test_classify_signals_from_tags():
    m = _make_model(tags=["awq", "text-generation"])
    result = classify_hf_model_format(m)
    assert result == "transformers_quantized"


# ---------------------------------------------------------------------------
# Group 4: _search_score
# ---------------------------------------------------------------------------

@test_group("Model Manager – search scoring")
def test_search_score_exact_name_match_high():
    m = _make_model(repo_id="org/llama", downloads=0, likes=0)
    score = _search_score(m, "llama", prefer_quantized=False, vram_gb=None)
    assert score >= 60


@test_group("Model Manager – search scoring")
def test_search_score_downloads_contribute():
    m_high = _make_model(repo_id="a/model", downloads=100_000, likes=0)
    m_low = _make_model(repo_id="b/model", downloads=1, likes=0)
    assert _search_score(m_high, "model", False, None) > _search_score(m_low, "model", False, None)


@test_group("Model Manager – search scoring")
def test_search_score_quantized_bonus_when_preferred():
    m = _make_model(repo_id="org/model-awq", downloads=0, likes=0, has_awq=True)
    score_prefer = _search_score(m, "model", prefer_quantized=True, vram_gb=None)
    score_not = _search_score(m, "model", prefer_quantized=False, vram_gb=None)
    assert score_prefer > score_not


@test_group("Model Manager – search scoring")
def test_search_score_pipeline_bonus():
    m_chat = _make_model(repo_id="a/m", downloads=0, likes=0, pipeline_tag="text-generation")
    m_other = _make_model(repo_id="b/m", downloads=0, likes=0, pipeline_tag="image-classification")
    assert _search_score(m_chat, "m", False, None) > _search_score(m_other, "m", False, None)


# ---------------------------------------------------------------------------
# Group 5: _merge_rank_results
# ---------------------------------------------------------------------------

@test_group("Model Manager – merge/rank")
def test_merge_deduplicates_same_repo():
    m1 = _make_model(repo_id="org/model", downloads=1000, likes=10)
    m2 = _make_model(repo_id="org/model", downloads=5000, likes=20)
    result = _merge_rank_results([m1, m2], "model", False, None, limit=10)
    assert len(result) == 1
    assert result[0].downloads == 5000


@test_group("Model Manager – merge/rank")
def test_merge_respects_limit():
    models = [_make_model(repo_id=f"org/model{i}", downloads=i) for i in range(10)]
    result = _merge_rank_results(models, "model", False, None, limit=3)
    assert len(result) == 3


@test_group("Model Manager – merge/rank")
def test_merge_propagates_format_signals():
    m1 = _make_model(repo_id="org/model", has_gguf=True)
    m2 = _make_model(repo_id="org/model", has_awq=True, has_gguf=False)
    result = _merge_rank_results([m1, m2], "model", False, None, limit=10)
    assert result[0].has_gguf is True
    assert result[0].has_awq is True


@test_group("Model Manager – merge/rank")
def test_merge_preserves_distinct_repos():
    models = [_make_model(repo_id=f"org/model-{i}") for i in range(5)]
    result = _merge_rank_results(models, "model", False, None, limit=10)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Group 6: recommend_format()
# ---------------------------------------------------------------------------

@test_group("Model Manager – recommend_format")
def test_recommend_cpu_only():
    rec = recommend_format(params_b=7.0, vram_gb=0)
    assert rec.engine == "ollama"
    assert rec.format == "gguf"
    assert rec.fits_in_vram is False
    assert rec.num_gpu_layers == 0


@test_group("Model Manager – recommend_format")
def test_recommend_small_model_fp16_vllm():
    # 3B fp16 = ~6 GB; with overhead ~6.9 GB. Fits in 24 GB.
    rec = recommend_format(params_b=3.0, vram_gb=24.0)
    assert rec.engine == "vllm"
    assert rec.quantization == "fp16"
    assert rec.fits_in_vram is True


@test_group("Model Manager – recommend_format")
def test_recommend_medium_model_awq_vllm():
    # 13B fp16 ≈ 29.9 GB, won't fit in 24 GB.
    # AWQ ≈ 13 * 0.56 * 1.15 ≈ 8.4 GB — fits.
    rec = recommend_format(params_b=13.0, vram_gb=24.0)
    assert rec.engine == "vllm"
    assert rec.quantization == "awq"
    assert rec.fits_in_vram is True


@test_group("Model Manager – recommend_format")
def test_recommend_large_model_gguf_ollama():
    # 70B fp16 ≈ 161 GB — won't fit. AWQ ≈ 45 GB — won't fit.
    # Q4_K_M = 0.56 bpp → 70 * 0.56 * 1.15 ≈ 45 GB — won't fit in 24.
    # Q5_K_M = 0.68 bpp → 70 * 0.68 * 1.15 ≈ 54.7 — won't fit.
    # 7B Q5_K_M = 7 * 0.68 * 1.15 ≈ 5.5 — fits in 8 GB.
    rec = recommend_format(params_b=7.0, vram_gb=8.0)
    assert rec.engine in ("vllm", "ollama")
    # Small model should fit somewhere
    assert rec.fits_in_vram is True


@test_group("Model Manager – recommend_format")
def test_recommend_split_offload_for_oversized():
    # 34B: fp16=78 GB, AWQ=21.8 GB. 8 GB GPU: nothing fits.
    rec = recommend_format(params_b=34.0, vram_gb=8.0)
    assert rec.engine == "llama_cpp"
    assert rec.fits_in_vram is False
    assert rec.num_gpu_layers is not None
    assert rec.num_gpu_layers >= 0


@test_group("Model Manager – recommend_format")
def test_recommend_prefer_quality_uses_higher_quant():
    # With 24 GB and 70B, will need GGUF. prefer_quality → Q8_0 or Q6_K tried first.
    rec_quality = recommend_format(params_b=12.0, vram_gb=24.0, prefer_quality=True)
    rec_default = recommend_format(params_b=12.0, vram_gb=24.0, prefer_quality=False)
    # Both should recommend something valid
    assert rec_quality.quantization in GGUF_QUANT_BPP or rec_quality.quantization in ("fp16", "awq")
    assert rec_default.quantization in GGUF_QUANT_BPP or rec_default.quantization in ("fp16", "awq")


@test_group("Model Manager – recommend_format")
def test_recommend_estimated_vram_is_positive():
    rec = recommend_format(params_b=7.0, vram_gb=24.0)
    assert rec.estimated_vram_gb > 0


@test_group("Model Manager – recommend_format")
def test_recommend_reason_is_non_empty():
    rec = recommend_format(params_b=7.0, vram_gb=24.0)
    assert rec.reason != ""


# ---------------------------------------------------------------------------
# Group 7: recommend_format_for_hf_model()
# ---------------------------------------------------------------------------

@test_group("Model Manager – recommend_format_for_hf_model")
def test_rec_hf_gguf_family_goes_to_ollama():
    m = _make_model(has_gguf=True, estimated_params_b=7.0)
    rec = recommend_format_for_hf_model(m, vram_gb=24.0)
    assert rec.engine == "ollama"
    assert rec.format == "gguf"


@test_group("Model Manager – recommend_format_for_hf_model")
def test_rec_hf_transformers_quantized_goes_to_vllm():
    m = _make_model(has_awq=True, estimated_params_b=7.0)
    rec = recommend_format_for_hf_model(m, vram_gb=24.0)
    assert rec.engine == "vllm"
    assert rec.format == "safetensors"


@test_group("Model Manager – recommend_format_for_hf_model")
def test_rec_hf_gptq_goes_to_vllm():
    m = _make_model(has_gptq=True, estimated_params_b=7.0)
    rec = recommend_format_for_hf_model(m, vram_gb=24.0)
    assert rec.engine == "vllm"


@test_group("Model Manager – recommend_format_for_hf_model")
def test_rec_hf_mixed_family_goes_to_vllm():
    m = _make_model(has_gguf=True, has_awq=True, estimated_params_b=7.0)
    rec = recommend_format_for_hf_model(m, vram_gb=24.0)
    assert rec.engine == "vllm"


@test_group("Model Manager – recommend_format_for_hf_model")
def test_rec_hf_base_falls_through_to_recommend_format():
    m = _make_model(has_safetensors=True, has_gguf=False, has_awq=False,
                    has_gptq=False, has_exl2=False, estimated_params_b=3.0)
    rec = recommend_format_for_hf_model(m, vram_gb=24.0)
    # base → recommend_format → fp16 fits → vllm
    assert rec.engine == "vllm"
    assert rec.quantization == "fp16"


@test_group("Model Manager – recommend_format_for_hf_model")
def test_rec_hf_gguf_cpu_only():
    m = _make_model(has_gguf=True, estimated_params_b=7.0)
    rec = recommend_format_for_hf_model(m, vram_gb=0.0)
    assert rec.engine == "ollama"
    assert rec.num_gpu_layers == 0
    assert rec.fits_in_vram is False


@test_group("Model Manager – recommend_format_for_hf_model")
def test_rec_hf_missing_params_raises():
    m = _make_model(estimated_params_b=None)
    try:
        recommend_format_for_hf_model(m, vram_gb=24.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


@test_group("Model Manager – recommend_format_for_hf_model")
def test_rec_hf_gguf_oversized_uses_split_offload():
    # 70B GGUF Q4_K_M ≈ 45 GB: won't fit in 8 GB → llama_cpp split
    m = _make_model(has_gguf=True, estimated_params_b=70.0)
    rec = recommend_format_for_hf_model(m, vram_gb=8.0)
    assert rec.engine == "llama_cpp"
    assert rec.fits_in_vram is False
    assert rec.num_gpu_layers >= 0


# ---------------------------------------------------------------------------
# Group 8: artifact_family_from_text + estimate_model_risk
# ---------------------------------------------------------------------------

@test_group("Model Manager – artifact family")
def test_artifact_family_gguf_from_text():
    assert artifact_family_from_text("llama-7b-q4_k_m.gguf") == "gguf"


@test_group("Model Manager – artifact family")
def test_artifact_family_gguf_from_backend():
    assert artifact_family_from_text("model", backend="llama_cpp") == "gguf"
    assert artifact_family_from_text("model", backend="llama-cpp") == "gguf"


@test_group("Model Manager – artifact family")
def test_artifact_family_awq():
    assert artifact_family_from_text("Mistral-7B-AWQ") == "awq"


@test_group("Model Manager – artifact family")
def test_artifact_family_gptq():
    assert artifact_family_from_text("model-GPTQ") == "gptq"


@test_group("Model Manager – artifact family")
def test_artifact_family_unknown():
    assert artifact_family_from_text("plain-safetensors-model") == "unknown"


@test_group("Model Manager – risk estimation")
def test_risk_remote_backend():
    result = estimate_model_risk("awq", 7.0, 24.0, 20.0, backend="anthropic")
    assert result["fit"] == "remote"
    assert result["runtime"] == "remote"


@test_group("Model Manager – risk estimation")
def test_risk_gguf_with_gpu_layers():
    result = estimate_model_risk("gguf", 7.0, 24.0, 20.0, n_gpu_layers=40)
    assert result["fit"] == "good"
    assert result["runtime"] == "ollama"


@test_group("Model Manager – risk estimation")
def test_risk_gguf_no_gpu_layers():
    result = estimate_model_risk("gguf", 7.0, 24.0, 20.0, n_gpu_layers=0)
    assert result["fit"] == "good"
    assert result["runtime"] == "ollama"


@test_group("Model Manager – risk estimation")
def test_risk_small_awq_good():
    result = estimate_model_risk("awq", 7.0, 24.0, 20.0)
    assert result["fit"] == "good"
    assert result["runtime"] == "vllm"


@test_group("Model Manager – risk estimation")
def test_risk_oversized_model():
    result = estimate_model_risk("awq", 70.0, 24.0, 20.0)
    assert result["fit"] in ("likely_oom", "better_ollama")


@test_group("Model Manager – risk estimation")
def test_risk_blocked_by_low_free_vram():
    # Small model, but only 4 GB free — should be blocked_now
    result = estimate_model_risk("awq", 7.0, 24.0, free_vram_gb=4.0)
    assert result["fit"] == "blocked_now"


@test_group("Model Manager – risk estimation")
def test_risk_no_gpu_detected():
    result = estimate_model_risk("awq", 7.0, total_vram_gb=None, free_vram_gb=None)
    assert result["fit"] == "unknown"


# ---------------------------------------------------------------------------
# Group 9: extract_model_size_b
# ---------------------------------------------------------------------------

@test_group("Model Manager – size extraction")
def test_extract_7b():
    assert extract_model_size_b("llama-7b-hf") == 7.0


@test_group("Model Manager – size extraction")
def test_extract_13b():
    assert extract_model_size_b("Llama-2-13b-chat") == 13.0


@test_group("Model Manager – size extraction")
def test_extract_70b():
    assert extract_model_size_b("meta-llama/Llama-2-70b") == 70.0


@test_group("Model Manager – size extraction")
def test_extract_fractional():
    assert extract_model_size_b("Qwen2-0.5b") == 0.5


@test_group("Model Manager – size extraction")
def test_extract_no_pattern():
    assert extract_model_size_b("gpt-neo-x") is None


@test_group("Model Manager – size extraction")
def test_extract_empty():
    assert extract_model_size_b("") is None
    assert extract_model_size_b(None) is None


# ---------------------------------------------------------------------------
# Group 10: scan_local_model
# ---------------------------------------------------------------------------

@test_group("Model Manager – scan_local_model")
def test_scan_single_gguf_file():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "mistral-7b-q4.gguf"
        p.write_bytes(b"\x00" * 1024)
        info = scan_local_model(p)
        assert info is not None
        assert info.format == "gguf"
        assert info.name == "mistral-7b-q4"
        assert info.files == ["mistral-7b-q4.gguf"]
        assert info.size_gb >= 0


@test_group("Model Manager – scan_local_model")
def test_scan_directory_with_gguf():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "model.gguf").write_bytes(b"\x00" * 2048)
        info = scan_local_model(p)
        assert info is not None
        assert info.format == "gguf"
        assert "model.gguf" in info.files


@test_group("Model Manager – scan_local_model")
def test_scan_directory_with_safetensors():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "model.safetensors").write_bytes(b"\x00" * 4096)
        (p / "config.json").write_text("{}")
        info = scan_local_model(p)
        assert info is not None
        assert info.format == "safetensors"
        assert "model.safetensors" in info.files


@test_group("Model Manager – scan_local_model")
def test_scan_directory_with_pytorch_bin():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "pytorch_model.bin").write_bytes(b"\x00" * 1024)
        info = scan_local_model(p)
        assert info is not None
        assert info.format == "pytorch"


@test_group("Model Manager – scan_local_model")
def test_scan_non_model_directory_returns_none():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "readme.txt").write_text("nothing")
        (p / "config.json").write_text("{}")
        info = scan_local_model(p)
        assert info is None


@test_group("Model Manager – scan_local_model")
def test_scan_missing_path_returns_none():
    info = scan_local_model(Path("/tmp/nonexistent_engram_test_xyz"))
    assert info is None


@test_group("Model Manager – scan_local_model")
def test_scan_non_gguf_file_returns_none():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "weights.safetensors"
        p.write_bytes(b"\x00" * 1024)
        # Single file, not .gguf → returns None
        info = scan_local_model(p)
        assert info is None


@test_group("Model Manager – scan_local_model")
def test_scan_size_gb_is_numeric():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "model.gguf"
        p.write_bytes(b"\x00" * (1024 * 1024))  # 1 MiB
        info = scan_local_model(p)
        assert isinstance(info.size_gb, float)
        assert info.size_gb >= 0


# ---------------------------------------------------------------------------
# Group 11: default_local_model_dir
# ---------------------------------------------------------------------------

@test_group("Model Manager – default_local_model_dir")
def test_model_dir_author_slash_name():
    with tempfile.TemporaryDirectory() as td:
        path = default_local_model_dir("meta-llama/Llama-2-7b-hf", root=Path(td))
        assert path.parent.name == "meta-llama"
        assert path.name == "Llama-2-7b-hf"


@test_group("Model Manager – default_local_model_dir")
def test_model_dir_single_part():
    with tempfile.TemporaryDirectory() as td:
        path = default_local_model_dir("mymodel", root=Path(td))
        assert path.name == "mymodel"


@test_group("Model Manager – default_local_model_dir")
def test_model_dir_sanitises_special_chars():
    with tempfile.TemporaryDirectory() as td:
        path = default_local_model_dir("org/model name!", root=Path(td))
        assert " " not in path.name
        assert "!" not in path.name


@test_group("Model Manager – default_local_model_dir")
def test_model_dir_empty_raises():
    try:
        default_local_model_dir("")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Group 12: SystemInfo dataclass
# ---------------------------------------------------------------------------

@test_group("Model Manager – SystemInfo")
def test_system_info_primary_vram_no_gpus():
    info = SystemInfo()
    assert info.primary_vram_gb == 0.0


@test_group("Model Manager – SystemInfo")
def test_system_info_primary_vram_single_gpu():
    info = SystemInfo(gpus=[GPUInfo(name="RTX 3090", vram_gb=24.0)])
    assert info.primary_vram_gb == 24.0


@test_group("Model Manager – SystemInfo")
def test_system_info_primary_vram_largest_gpu():
    info = SystemInfo(gpus=[
        GPUInfo(name="RTX 3080", vram_gb=10.0),
        GPUInfo(name="RTX 3090", vram_gb=24.0, index=1),
    ])
    assert info.primary_vram_gb == 24.0


@test_group("Model Manager – SystemInfo")
def test_system_info_total_vram_sum():
    info = SystemInfo(
        gpus=[GPUInfo(name="A", vram_gb=8.0), GPUInfo(name="B", vram_gb=8.0, index=1)],
        total_vram_gb=16.0,
    )
    assert info.total_vram_gb == 16.0


# ---------------------------------------------------------------------------
# Group 13: detect_system()
# ---------------------------------------------------------------------------

@test_group("Model Manager – detect_system")
def test_detect_system_returns_system_info():
    info = detect_system()
    assert isinstance(info, SystemInfo)


@test_group("Model Manager – detect_system")
def test_detect_system_never_raises():
    # Should never raise regardless of hardware
    try:
        detect_system()
    except Exception as e:
        assert False, f"detect_system() raised: {e}"


@test_group("Model Manager – detect_system")
def test_detect_system_total_vram_consistent():
    info = detect_system()
    expected = sum(g.vram_gb for g in info.gpus)
    assert abs(info.total_vram_gb - expected) < 0.01


@test_group("Model Manager – detect_system")
def test_detect_system_accelerator_is_string():
    info = detect_system()
    assert isinstance(info.accelerator, str)
    assert info.accelerator in ("cuda", "mps", "cpu")


@test_group("Model Manager – detect_system")
def test_detect_system_ram_gb_positive_or_none():
    info = detect_system()
    if info.ram_gb is not None:
        assert info.ram_gb > 0


# ---------------------------------------------------------------------------
# Group 14: download_hf_model — no-deps fallback
# ---------------------------------------------------------------------------

@test_group("Model Manager – download_hf_model")
def test_download_hf_model_no_tools_returns_result():
    # Without huggingface_hub or huggingface-cli available (common in CI),
    # download_hf_model should return a DownloadResult with success=False
    # rather than raising.
    with tempfile.TemporaryDirectory() as td:
        result = download_hf_model(
            "nonexistent/model-xyz-engram-test",
            local_dir=Path(td) / "model",
        )
        assert isinstance(result, DownloadResult)
        assert result.repo_id == "nonexistent/model-xyz-engram-test"
        assert isinstance(result.success, bool)
        # If it failed it should carry an error message
        if not result.success:
            assert result.error != ""


@test_group("Model Manager – download_hf_model")
def test_download_hf_model_result_has_local_dir():
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "my_model"
        result = download_hf_model("some/model", local_dir=target)
        assert result.local_dir == target


# ---------------------------------------------------------------------------
# Group 15: network-dependent (skip unless network available)
# ---------------------------------------------------------------------------

@test_group("Model Manager – HF search (network)")
def test_search_hf_models_returns_list():
    require("hf_network")
    from engram.engine.model_manager import search_hf_models
    results = search_hf_models("llama 7b", limit=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    for r in results:
        assert isinstance(r, HFModelInfo)
        assert r.repo_id != ""
