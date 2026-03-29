from engram.engine.model_manager import HFModelInfo, _merge_rank_results


def test_merge_rank_results_prefers_quantized_match_for_low_vram():
    base = HFModelInfo(
        repo_id="Qwen/Qwen3.5-35B-A3B",
        name="Qwen3.5-35B-A3B",
        downloads=1000,
        likes=100,
        pipeline_tag="",
        tags=[],
        estimated_params_b=35.0,
        estimated_size_gb_fp16=70.0,
    )
    awq = HFModelInfo(
        repo_id="QuantTrio/Qwen3.5-35B-A3B-AWQ",
        name="Qwen3.5-35B-A3B-AWQ",
        downloads=100,
        likes=10,
        pipeline_tag="",
        tags=["awq"],
        has_safetensors=True,
        estimated_params_b=35.0,
        estimated_size_gb_fp16=70.0,
    )

    ranked = _merge_rank_results([base, awq], "Qwen3.5", True, 24.0, 10)
    assert ranked[0].repo_id == "QuantTrio/Qwen3.5-35B-A3B-AWQ"
