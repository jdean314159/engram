from engram.engine.model_manager import (
    HFModelInfo,
    classify_hf_model_format,
    recommend_format_for_hf_model,
)


def test_gptq_result_prefers_vllm_not_ollama():
    model = HFModelInfo(
        repo_id="Qwen3.5/Qwen3.5-32B-A3B-GPTQ-Int4",
        name="Qwen3.5-32B-A3B-GPTQ-Int4",
        tags=["gptq", "text-generation"],
        has_safetensors=True,
        estimated_params_b=32.0,
        estimated_size_gb_fp16=64.0,
    )

    assert classify_hf_model_format(model) == "transformers_quantized"
    rec = recommend_format_for_hf_model(model, 24.0)
    assert rec.engine == "vllm"
    assert rec.quantization == "gptq_int4"



def test_mixed_format_result_is_marked_ambiguous_but_still_prefers_transformers_quant():
    model = HFModelInfo(
        repo_id="SomeOrg/qwen3_5_35b_a3b_gptq_int4_q3_k_m",
        name="qwen3_5_35b_a3b_gptq_int4_q3_k_m",
        tags=["gptq", "gguf"],
        has_gguf=True,
        has_safetensors=True,
        estimated_params_b=35.0,
        estimated_size_gb_fp16=70.0,
    )

    assert classify_hf_model_format(model) == "mixed"
    rec = recommend_format_for_hf_model(model, 24.0)
    assert rec.engine == "vllm"
    assert "mixed" in rec.reason.lower()



def test_gguf_result_prefers_ollama_even_if_dense_weights_would_fit():
    model = HFModelInfo(
        repo_id="SomeOrg/TinyModel-GGUF",
        name="TinyModel-GGUF",
        tags=["gguf"],
        has_gguf=True,
        estimated_params_b=3.0,
        estimated_size_gb_fp16=6.0,
    )

    assert classify_hf_model_format(model) == "gguf"
    rec = recommend_format_for_hf_model(model, 24.0)
    assert rec.engine == "ollama"
    assert rec.format == "gguf"


def test_gguf_partial_offload_recommends_llama_cpp():
    """GGUF model that exceeds VRAM should recommend llama_cpp with n_gpu_layers."""
    model = HFModelInfo(
        repo_id="bartowski/Qwen2.5-32B-Instruct-GGUF",
        name="Qwen2.5-32B-Instruct-GGUF",
        tags=["gguf"],
        has_gguf=True,
        estimated_params_b=32.0,
        estimated_size_gb_fp16=64.0,
    )
    # 8GB VRAM — Q4_K_M (20.6GB) won't fit; expect llama_cpp split
    rec = recommend_format_for_hf_model(model, 8.0)
    assert rec.engine == "llama_cpp"
    assert rec.format == "gguf"
    assert rec.num_gpu_layers is not None and rec.num_gpu_layers > 0
    assert "llama" in rec.reason.lower() or "split" in rec.reason.lower()


def test_recommend_format_partial_offload_returns_llama_cpp():
    """recommend_format() partial-offload path returns llama_cpp, not ollama."""
    from engram.engine.model_manager import recommend_format
    # 35B model, 8GB VRAM — nothing fits at fp16 or AWQ; GGUF also too big
    rec = recommend_format(35.0, 8.0)
    assert rec.engine == "llama_cpp"
    assert rec.num_gpu_layers is not None
    assert rec.num_gpu_layers >= 0


def test_recommend_format_cpu_only_returns_ollama():
    """CPU-only (vram=0) still returns Ollama for simple deployments."""
    from engram.engine.model_manager import recommend_format
    rec = recommend_format(7.0, 0.0)
    assert rec.engine == "ollama"
    assert rec.num_gpu_layers == 0


def test_format_recommendation_has_gguf_path_field():
    """FormatRecommendation has the gguf_path field."""
    from engram.engine.model_manager import FormatRecommendation
    rec = FormatRecommendation(
        engine="llama_cpp", format="gguf", quantization="Q4_K_M",
        fits_in_vram=False, estimated_vram_gb=6.0,
        num_gpu_layers=21, gguf_path="/models/test.gguf",
        reason="test",
    )
    assert rec.gguf_path == "/models/test.gguf"


def test_register_model_in_config_llama_cpp(tmp_path):
    """register_model_in_config writes n_gpu_layers for llama_cpp engines."""
    import yaml
    from engram.engine.model_manager import register_model_in_config
    config_path = tmp_path / "engines.yaml"

    register_model_in_config(
        engine_name="test_llama",
        engine_type="llama_cpp",
        model_id="test_llama",
        config_path=config_path,
        num_gpu=40,
        extra={"gguf_path": "/models/q.gguf"},
    )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    entry = cfg["engines"]["test_llama"]
    assert entry["type"] == "llama_cpp"
    assert entry["n_gpu_layers"] == 40
    assert entry["gguf_path"] == "/models/q.gguf"
    assert "127.0.0.1:8080" in entry.get("base_url", "")
    assert entry.get("timeout", 0) == 300
