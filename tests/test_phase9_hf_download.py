import sys
from pathlib import Path

from engram.engine.model_manager import default_local_model_dir, download_hf_model


def test_default_local_model_dir_uses_engram_models_root(tmp_path):
    path = default_local_model_dir("QuantTrio/Qwen3.5-35B-A3B-AWQ", root=tmp_path)
    assert path == tmp_path / "QuantTrio" / "Qwen3.5-35B-A3B-AWQ"


def test_download_hf_model_uses_huggingface_hub_when_available(monkeypatch, tmp_path):
    calls = {}

    class FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            calls.update(kwargs)
            Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(kwargs["local_dir"]) / "config.json").write_text("{}", encoding="utf-8")
            return kwargs["local_dir"]

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)

    target = tmp_path / "QuantTrio" / "Qwen3.5-35B-A3B-AWQ"
    result = download_hf_model("QuantTrio/Qwen3.5-35B-A3B-AWQ", local_dir=target)

    assert result.success is True
    assert result.local_dir == target
    assert calls["repo_id"] == "QuantTrio/Qwen3.5-35B-A3B-AWQ"
    assert Path(calls["local_dir"]) == target
