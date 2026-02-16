# tests/pipeline/test_orchestrator.py
import pytest
from unittest.mock import MagicMock
from backend.pipeline.orchestrator import PipelineConfig, run_pipeline

FIXTURE = "tests/fixtures/test_2s.mp4"


class TestPipelineConfig:
    def test_default_config(self):
        cfg = PipelineConfig(
            video_path=FIXTURE,
            attack_method="fgsm",
            target_text="an empty room",
        )
        assert cfg.video_path == FIXTURE
        assert cfg.attack_method == "fgsm"


class TestRunPipeline:
    def test_full_pipeline_preview(self):
        cfg = PipelineConfig(
            video_path=FIXTURE,
            attack_method="fgsm",
            preset="preview",
            target_text="a sunset over the ocean",
            clip_labels=["a number", "a sunset over the ocean", "a blank screen"],
        )
        progress_calls = []
        def on_progress(pct, stage):
            progress_calls.append((pct, stage))

        result = run_pipeline(cfg, on_progress=on_progress)

        assert "output_path" in result
        assert "original_classifications" in result
        assert "perturbed_classifications" in result
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 100
