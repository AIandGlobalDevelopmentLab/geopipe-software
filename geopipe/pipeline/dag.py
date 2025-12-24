"""DAG-based pipeline construction and execution."""

from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from geopipe.pipeline.tasks import Task, task as task_decorator
from geopipe.fusion.schema import FusionSchema


console = Console()


class Pipeline:
    """
    DAG-based pipeline for executing a sequence of tasks.

    A Pipeline manages the execution of tasks in order, with support for
    checkpointing, resumption, and different execution backends.

    Attributes:
        name: Pipeline identifier
        tasks: List of tasks in execution order
        checkpoint_dir: Directory for pipeline checkpoints

    Example:
        >>> pipeline = Pipeline([
        ...     download_imagery,
        ...     extract_features,
        ...     run_model,
        ... ])
        >>> result = pipeline.run(resume=True)
    """

    def __init__(
        self,
        tasks: list[Task | Callable] | None = None,
        name: str = "geopipe_pipeline",
        checkpoint_dir: str = ".geopipe/pipeline",
    ) -> None:
        """
        Initialize a pipeline.

        Args:
            tasks: List of Task objects or callables
            name: Pipeline identifier
            checkpoint_dir: Directory for pipeline state
        """
        self.name = name
        self.checkpoint_dir = Path(checkpoint_dir)
        self._tasks: list[Task] = []

        if tasks:
            for t in tasks:
                self.add_stage(t)

    def add_stage(
        self,
        func: Task | Callable,
        cache: bool = True,
        checkpoint: bool = True,
        **kwargs: Any,
    ) -> "Pipeline":
        """
        Add a stage to the pipeline.

        Args:
            func: Task or callable to add
            cache: Enable caching for this stage
            checkpoint: Enable checkpointing for this stage
            **kwargs: Additional task configuration

        Returns:
            Self for chaining
        """
        if isinstance(func, Task):
            self._tasks.append(func)
        else:
            # Wrap in Task
            wrapped = task_decorator(cache=cache, checkpoint=checkpoint, **kwargs)(func)
            self._tasks.append(wrapped)

        return self

    @classmethod
    def from_schema(cls, schema: FusionSchema, **kwargs: Any) -> "Pipeline":
        """
        Create a pipeline from a FusionSchema.

        The resulting pipeline has a single stage that executes the fusion.

        Args:
            schema: FusionSchema to convert
            **kwargs: Additional Pipeline configuration

        Returns:
            Pipeline instance
        """
        pipeline = cls(name=schema.name, **kwargs)

        @task_decorator(cache=True, checkpoint=True)
        def execute_fusion():
            return schema.execute(show_progress=False)

        pipeline.add_stage(execute_fusion)
        return pipeline

    def run(
        self,
        initial_input: Any = None,
        resume: bool = True,
        executor: Any = None,
        show_progress: bool = True,
    ) -> Any:
        """
        Execute the pipeline.

        Args:
            initial_input: Input to first stage
            resume: Whether to resume from checkpoints
            executor: Optional executor for distributed execution
            show_progress: Whether to show progress bar

        Returns:
            Output of final stage
        """
        if not self._tasks:
            raise ValueError("Pipeline has no stages")

        if executor is not None:
            return self._run_distributed(initial_input, executor)

        # Determine starting point for resume
        start_idx = 0
        if resume:
            start_idx = self._find_resume_point()
            if start_idx > 0:
                console.print(f"[yellow]Resuming from stage {start_idx + 1}[/yellow]")

        # Execute stages
        result = initial_input

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task_id = progress.add_task(
                    "Running pipeline...",
                    total=len(self._tasks) - start_idx,
                )

                for i, stage in enumerate(self._tasks[start_idx:], start=start_idx):
                    progress.update(task_id, description=f"Stage {i + 1}: {stage.name}")

                    if result is None:
                        result = stage()
                    else:
                        result = stage(result)

                    self._save_pipeline_state(i)
                    progress.advance(task_id)
        else:
            for i, stage in enumerate(self._tasks[start_idx:], start=start_idx):
                console.print(f"Stage {i + 1}/{len(self._tasks)}: {stage.name}")

                if result is None:
                    result = stage()
                else:
                    result = stage(result)

                self._save_pipeline_state(i)

        console.print("[green]Pipeline completed successfully[/green]")
        return result

    def _run_distributed(self, initial_input: Any, executor: Any) -> Any:
        """Execute pipeline using a distributed executor."""
        # Delegate to executor
        return executor.run_pipeline(self, initial_input)

    def _find_resume_point(self) -> int:
        """Find the stage to resume from based on checkpoints."""
        state_file = self.checkpoint_dir / f"{self.name}_state.json"

        if not state_file.exists():
            return 0

        import json

        with open(state_file) as f:
            state = json.load(f)

        last_completed = state.get("last_completed_stage", -1)
        return min(last_completed + 1, len(self._tasks))

    def _save_pipeline_state(self, completed_stage: int) -> None:
        """Save pipeline execution state."""
        import json
        from datetime import datetime

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state_file = self.checkpoint_dir / f"{self.name}_state.json"

        state = {
            "name": self.name,
            "last_completed_stage": completed_stage,
            "total_stages": len(self._tasks),
            "timestamp": datetime.now().isoformat(),
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def clear_checkpoints(self) -> None:
        """Clear all pipeline checkpoints."""
        # Clear pipeline state
        state_file = self.checkpoint_dir / f"{self.name}_state.json"
        if state_file.exists():
            state_file.unlink()

        # Clear task checkpoints
        for stage in self._tasks:
            stage.clear_checkpoints()

        console.print(f"[yellow]Cleared checkpoints for {self.name}[/yellow]")

    def visualize(self) -> str:
        """Generate a text visualization of the pipeline DAG."""
        lines = [f"Pipeline: {self.name}", "=" * (len(self.name) + 10)]

        for i, stage in enumerate(self._tasks):
            prefix = "├──" if i < len(self._tasks) - 1 else "└──"
            resources = stage.config.resources
            res_str = f" [{resources}]" if resources else ""
            lines.append(f"  {prefix} [{i + 1}] {stage.name}{res_str}")

        return "\n".join(lines)

    @property
    def stages(self) -> list[str]:
        """Return list of stage names."""
        return [t.name for t in self._tasks]

    def __len__(self) -> int:
        return len(self._tasks)

    def __repr__(self) -> str:
        return f"Pipeline({self.name!r}, stages={len(self._tasks)})"
