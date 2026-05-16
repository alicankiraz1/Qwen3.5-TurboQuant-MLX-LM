from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_covers_preview_artifacts():
    content = (ROOT / ".gitignore").read_text()
    for needle in (
        ".venv/",
        ".venv312/",
        ".pytest_cache/",
        "__pycache__/",
        ".DS_Store",
        "__MACOSX/",
        "references/*.pdf",
    ):
        assert needle in content


def test_pyproject_pins_tested_mlx_minor_and_excludes_reference_artifacts():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    mlx_optional = payload["project"]["optional-dependencies"]["mlx"]
    assert "mlx>=0.31.2,<0.32" in mlx_optional
    assert "mlx-lm>=0.31.3,<0.32" in mlx_optional

    sdist_excludes = payload["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert "/.venv" in sdist_excludes
    assert "/.venv312" in sdist_excludes
    assert "/references/*.pdf" in sdist_excludes


def test_pyproject_declares_serialize_extra_for_safetensors():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    serialize_optional = payload["project"]["optional-dependencies"]["serialize"]
    assert any(item.startswith("safetensors") for item in serialize_optional)


def test_pyproject_supports_numpy_2_and_modern_typer():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = payload["project"]["dependencies"]
    numpy_pin = next(item for item in deps if item.startswith("numpy"))
    typer_pin = next(item for item in deps if item.startswith("typer"))
    assert "<3" in numpy_pin, "numpy should permit 2.x in addition to 1.26+"
    assert ">=0.16" in typer_pin, "typer minimum should reflect modern Annotated support"


def test_export_preview_bundle_script_creates_clean_bundle(tmp_path):
    output_path = tmp_path / "preview-bundle.zip"
    script_path = ROOT / "scripts" / "export_preview_bundle.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--output", str(output_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert output_path.exists()
    assert result.stdout.strip().endswith("preview-bundle.zip")

    with zipfile.ZipFile(output_path) as archive:
        names = archive.namelist()

    assert "LICENSE" in names
    assert "README.md" in names
    assert "pyproject.toml" in names
    assert "src/turbomlx/cli.py" in names
    assert "docs/release_notes_v0.1_research_preview.md" in names
    assert all(".venv/" not in name for name in names)
    assert all(".venv312/" not in name for name in names)
    assert all(".pytest_cache/" not in name for name in names)
    assert all("__pycache__/" not in name for name in names)
    assert all("__MACOSX/" not in name for name in names)
    assert all(not name.startswith("references/") for name in names)
    assert all(not name.startswith("docs/superpowers/") for name in names)


def test_export_preview_bundle_script_can_export_clean_public_tree(tmp_path):
    output_dir = tmp_path / "public-tree"
    script_path = ROOT / "scripts" / "export_preview_bundle.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert output_dir.exists()
    assert str(output_dir) in result.stdout
    assert (output_dir / "LICENSE").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "pyproject.toml").exists()
    assert (output_dir / "src" / "turbomlx" / "cli.py").exists()
    assert (output_dir / "docs" / "release_notes_v0.1_research_preview.md").exists()
    assert not (output_dir / "docs" / "superpowers").exists()
    assert not (output_dir / ".venv").exists()
    assert not (output_dir / ".venv312").exists()
