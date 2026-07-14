"""Tests for yolo_learn.data.augment — augmentation presets."""

import pytest

from yolo_learn.data.augment import AUGMENT_PRESETS, get_augment_params, save_augment_config


class TestAugmentPresets:
    """增强预设测试."""

    def test_all_presets_exist(self):
        """所有预设都存在."""
        assert "none" in AUGMENT_PRESETS
        assert "light" in AUGMENT_PRESETS
        assert "medium" in AUGMENT_PRESETS
        assert "heavy" in AUGMENT_PRESETS

    def test_none_preset_is_empty(self):
        """none 预设为空 dict."""
        assert AUGMENT_PRESETS["none"] == {}

    def test_presets_are_progressive(self):
        """预设应逐渐增强（更多参数、更大值）."""
        assert len(AUGMENT_PRESETS["none"]) <= len(AUGMENT_PRESETS["light"])
        assert len(AUGMENT_PRESETS["light"]) <= len(AUGMENT_PRESETS["medium"])
        assert len(AUGMENT_PRESETS["medium"]) <= len(AUGMENT_PRESETS["heavy"])


class TestGetAugmentParams:
    """获取增强参数测试."""

    def test_valid_preset(self):
        """有效预设返回参数."""
        params = get_augment_params("light")
        assert isinstance(params, dict)
        assert "hsv_h" in params

    def test_unknown_preset_raises(self):
        """未知预设抛出 ValueError."""
        with pytest.raises(ValueError, match="未知的增强预设"):
            get_augment_params("nonexistent")

    def test_returns_copy(self):
        """返回副本，修改不影响原始预设."""
        params1 = get_augment_params("light")
        params1["hsv_h"] = 999
        params2 = get_augment_params("light")
        assert params2["hsv_h"] != 999


class TestSaveAugmentConfig:
    """保存增强配置测试."""

    def test_save_creates_file(self, tmp_path):
        """保存配置创建文件."""
        output = tmp_path / "augment.yaml"
        save_augment_config("medium", output)
        assert output.exists()

    def test_saved_content_is_valid_yaml(self, tmp_path):
        """保存的内容是有效的 YAML."""
        import yaml

        output = tmp_path / "augment.yaml"
        save_augment_config("light", output)
        with open(output) as f:
            data = yaml.safe_load(f)
        assert data["augment"] == "light"
        assert "params" in data
