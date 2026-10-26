"""
Smoke test suite for UMKM Digital Readiness package.
"""
from pathlib import Path
import yaml


def test_package_structure():
    """Verify that core package directories exist."""
    base_dir = Path(__file__).resolve().parent.parent
    assert (base_dir / "src" / "umkm").exists()
    assert (base_dir / "configs").exists()
    assert (base_dir / "data").exists()


def test_configs_loadable():
    """Verify YAML configuration files load properly."""
    base_dir = Path(__file__).resolve().parent.parent

    kecamatan_cfg = base_dir / "configs" / "kecamatan.yaml"
    assert kecamatan_cfg.exists()
    with open(kecamatan_cfg, "r", encoding="utf-8") as f:
        data_kec = yaml.safe_load(f)
    assert len(data_kec.get("kecamatan", [])) == 6

    kelas_cfg = base_dir / "configs" / "kelas.yaml"
    assert kelas_cfg.exists()
    with open(kelas_cfg, "r", encoding="utf-8") as f:
        data_kelas = yaml.safe_load(f)
    assert len(data_kelas.get("indikator", [])) == 4
    assert len(data_kelas.get("kelas_kematangan", [])) == 3
