from pathlib import Path

from engram import ProjectMemory, ProjectType


def test_project_memory_expands_tilde_base_dir(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    pm = ProjectMemory(
        project_id="default_project",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=Path("~/.engram"),
        llm_engine=None,
    )

    expected = fake_home / ".engram" / "default_project"
    assert Path(pm.get_stats()["project_dir"]) == expected
    assert expected.exists()

    literal_tilde_dir = Path.cwd() / "~"
    assert not literal_tilde_dir.exists()

    pm.close()
    
    
def test_project_memory_expands_tilde_telemetry_path(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("ENGRAM_TELEMETRY", "1")
    monkeypatch.setenv("ENGRAM_TELEMETRY_SINK", "jsonl")
    monkeypatch.setenv("ENGRAM_TELEMETRY_PATH", "~/.engram/telemetry.jsonl")

    pm = ProjectMemory(
        project_id="default_project",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=fake_home / ".engram",
        llm_engine=None,
    )

    sink = pm.telemetry.sink
    assert str(sink.path) == str((fake_home / ".engram" / "telemetry.jsonl").resolve())

    pm.close()
