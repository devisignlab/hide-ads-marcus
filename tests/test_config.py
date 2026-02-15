# tests/test_config.py
from backend.config import QualityPreset, get_preset, ATTACK_METHODS


class TestConfig:
    def test_preset_preview_exists(self):
        preset = get_preset("preview")
        assert preset.method == "fgsm"
        assert preset.keyframe_interval > 5

    def test_preset_standard_exists(self):
        preset = get_preset("standard")
        assert preset.method == "pgd"
        assert preset.pgd_steps >= 15

    def test_preset_high_exists(self):
        preset = get_preset("high")
        assert preset.method == "pgd"
        assert preset.keyframe_interval == 1

    def test_attack_methods_list(self):
        assert "fgsm" in ATTACK_METHODS
        assert "pgd" in ATTACK_METHODS
        assert "lsb" in ATTACK_METHODS
        assert "uap" in ATTACK_METHODS

    def test_invalid_preset_raises(self):
        import pytest
        with pytest.raises(KeyError):
            get_preset("nonexistent")
