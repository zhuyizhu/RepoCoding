from pathlib import Path

from app.engine.path_resolution import resolve_candidate_paths


def test_unique_module_prefix_is_resolved(tmp_path: Path):
    target = tmp_path / "module" / "src/main/java/example/Foo.java"
    target.parent.mkdir(parents=True)
    target.write_text("class Foo {}", encoding="utf-8")
    result, mappings = resolve_candidate_paths(
        [{"file_path": "src/main/java/example/Foo.java", "target": "src/main/java/example/Foo.java"}],
        tmp_path,
    )
    assert result[0]["file_path"] == "module/src/main/java/example/Foo.java"
    assert result[0]["native_file_path"] == "src/main/java/example/Foo.java"
    assert result[0]["target"] == result[0]["file_path"]
    assert len(mappings) == 1


def test_ambiguous_suffix_is_not_guessed(tmp_path: Path):
    for module in ("one", "two"):
        target = tmp_path / module / "src/main/java/example/Foo.java"
        target.parent.mkdir(parents=True)
        target.write_text("class Foo {}", encoding="utf-8")
    native = "src/main/java/example/Foo.java"
    result, mappings = resolve_candidate_paths([{"file_path": native}], tmp_path)
    assert result[0]["file_path"] == native
    assert mappings == []


def test_parent_traversal_is_not_resolved(tmp_path: Path):
    result, mappings = resolve_candidate_paths(
        [{"file_path": "../src/main/java/example/Foo.java"}], tmp_path
    )
    assert result[0]["file_path"].startswith("..")
    assert mappings == []
