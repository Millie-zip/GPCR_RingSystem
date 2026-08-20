from pathlib import Path

from gpcr_ringgen.preprocessing.paths import DATA_ROOT_ENV, data_root


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_data_root_uses_environment(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "production-data"
    monkeypatch.setenv(DATA_ROOT_ENV, str(configured))
    assert data_root() == configured.resolve()


def test_preprocessing_structure() -> None:
    package_root = PROJECT_ROOT / "src" / "gpcr_ringgen" / "preprocessing"
    script_root = PROJECT_ROOT / "scripts" / "preprocessing"
    assert {"ring_library", "global_scaffold", "pocket_prefilter", "diffusion_dataset"} <= {
        path.name for path in package_root.iterdir() if path.is_dir()
    }
    assert len(list((script_root / "source_ingestion").glob("*.py"))) == 5


def test_legacy_numbered_directories_are_removed() -> None:
    legacy = ("01_four_download", "02_fused-ring master library", "03_global scaffold", "04_pocket prefilter", "05_3d_diffusion_dataset")
    assert not any((PROJECT_ROOT / name).exists() for name in legacy)

