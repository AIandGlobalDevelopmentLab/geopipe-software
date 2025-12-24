"""Tests for pipeline and task functionality."""

import tempfile
from pathlib import Path

import pytest

from geopipe.pipeline.tasks import task, Task, TaskConfig
from geopipe.pipeline.dag import Pipeline


class TestTask:
    """Tests for Task decorator and class."""

    def test_basic_task(self):
        """Test basic task execution."""
        @task()
        def add_one(x):
            return x + 1

        result = add_one(5)
        assert result == 6

    def test_task_with_caching(self):
        """Test task with in-memory caching."""
        call_count = 0

        @task(cache=True)
        def expensive_fn(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_fn(10)
        assert result1 == 20
        assert call_count == 1

        # Second call with same args should use cache
        result2 = expensive_fn(10)
        assert result2 == 20
        assert call_count == 1  # Should not have called again

    def test_task_with_checkpoint(self, tmp_path):
        """Test task with disk checkpointing."""
        checkpoint_dir = str(tmp_path / "checkpoints")

        @task(checkpoint=True, checkpoint_dir=checkpoint_dir)
        def compute(x):
            return x ** 2

        # First call creates checkpoint
        result1 = compute(5)
        assert result1 == 25

        # Verify checkpoint file exists
        checkpoint_files = list(Path(checkpoint_dir).glob("*.pkl"))
        assert len(checkpoint_files) == 1

    def test_task_retries(self):
        """Test task retry logic."""
        attempts = 0

        @task(retries=2, retry_delay=0.01)
        def flaky_fn():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("Temporary failure")
            return "success"

        result = flaky_fn()
        assert result == "success"
        assert attempts == 3

    def test_task_retries_exhausted(self):
        """Test task failure after retries exhausted."""
        @task(retries=1, retry_delay=0.01)
        def always_fails():
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError, match="Always fails"):
            always_fails()

    def test_task_clear_cache(self):
        """Test clearing task cache."""
        @task(cache=True)
        def cached_fn(x):
            return x

        cached_fn(1)
        assert len(cached_fn._cache) == 1

        cached_fn.clear_cache()
        assert len(cached_fn._cache) == 0


class TestPipeline:
    """Tests for Pipeline class."""

    def test_basic_pipeline(self, tmp_path):
        """Test basic pipeline execution."""
        @task()
        def step1():
            return 10

        @task()
        def step2(x):
            return x * 2

        @task()
        def step3(x):
            return x + 5

        pipeline = Pipeline(
            [step1, step2, step3],
            name="test_pipeline",
            checkpoint_dir=str(tmp_path / "pipeline"),
        )
        result = pipeline.run(show_progress=False, resume=False)

        assert result == 25  # (10 * 2) + 5

    def test_pipeline_add_stage(self):
        """Test adding stages to pipeline."""
        def double(x):
            return x * 2

        pipeline = Pipeline(name="test")
        pipeline.add_stage(double)
        pipeline.add_stage(lambda x: x + 1)

        assert len(pipeline) == 2
        assert "double" in pipeline.stages

    def test_pipeline_with_initial_input(self, tmp_path):
        """Test pipeline with initial input."""
        @task()
        def add_ten(x):
            return x + 10

        pipeline = Pipeline(
            [add_ten],
            name="test",
            checkpoint_dir=str(tmp_path / "pipeline"),
        )
        result = pipeline.run(initial_input=5, show_progress=False, resume=False)

        assert result == 15

    def test_pipeline_visualization(self):
        """Test pipeline DAG visualization."""
        @task()
        def download():
            pass

        @task(resources={"memory": "16GB"})
        def process():
            pass

        @task()
        def save():
            pass

        pipeline = Pipeline([download, process, save], name="my_pipeline")
        viz = pipeline.visualize()

        assert "my_pipeline" in viz
        assert "download" in viz
        assert "process" in viz
        assert "save" in viz
        assert "16GB" in viz

    def test_pipeline_checkpointing(self, tmp_path):
        """Test pipeline state checkpointing."""
        checkpoint_dir = str(tmp_path / "pipeline")

        @task()
        def step1():
            return 1

        @task()
        def step2(x):
            return x + 1

        pipeline = Pipeline([step1, step2], name="test", checkpoint_dir=checkpoint_dir)
        result = pipeline.run(show_progress=False)

        assert result == 2

        # Check that state file was created
        state_file = Path(checkpoint_dir) / "test_state.json"
        assert state_file.exists()

    def test_empty_pipeline(self):
        """Test error on empty pipeline execution."""
        pipeline = Pipeline(name="empty")

        with pytest.raises(ValueError, match="no stages"):
            pipeline.run()

    def test_pipeline_stages_property(self):
        """Test stages property."""
        @task()
        def a():
            pass

        @task()
        def b():
            pass

        pipeline = Pipeline([a, b])
        assert pipeline.stages == ["a", "b"]
