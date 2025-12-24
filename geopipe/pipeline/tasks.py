"""Task decorator for pipeline stages with caching and checkpointing."""

import functools
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, TypeVar
from datetime import datetime

from rich.console import Console


console = Console()

F = TypeVar("F", bound=Callable[..., Any])


class TaskConfig:
    """Configuration for a pipeline task."""

    def __init__(
        self,
        cache: bool = False,
        checkpoint: bool = False,
        checkpoint_dir: str = ".geopipe/checkpoints",
        resources: dict[str, Any] | None = None,
        retries: int = 0,
        retry_delay: float = 1.0,
    ) -> None:
        self.cache = cache
        self.checkpoint = checkpoint
        self.checkpoint_dir = Path(checkpoint_dir)
        self.resources = resources or {}
        self.retries = retries
        self.retry_delay = retry_delay


class Task:
    """
    A pipeline task wrapping a function with caching and checkpointing.

    Tasks are the building blocks of pipelines. They wrap functions with
    optional caching (in-memory) and checkpointing (to disk).

    Attributes:
        func: The wrapped function
        config: Task configuration
        name: Task name (defaults to function name)

    Example:
        >>> @task(cache=True, checkpoint=True)
        ... def process_data(input_path):
        ...     return pd.read_csv(input_path)
    """

    def __init__(
        self,
        func: Callable[..., Any],
        config: TaskConfig,
        name: str | None = None,
    ) -> None:
        self.func = func
        self.config = config
        self.name = name or func.__name__
        self._cache: dict[str, Any] = {}

        functools.update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the task with caching/checkpointing."""
        cache_key = self._compute_cache_key(args, kwargs)

        # Check in-memory cache
        if self.config.cache and cache_key in self._cache:
            console.print(f"[dim]Cache hit for {self.name}[/dim]")
            return self._cache[cache_key]

        # Check disk checkpoint
        if self.config.checkpoint:
            checkpoint_path = self._get_checkpoint_path(cache_key)
            if checkpoint_path.exists():
                console.print(f"[dim]Resuming {self.name} from checkpoint[/dim]")
                return self._load_checkpoint(checkpoint_path)

        # Execute with retries
        result = self._execute_with_retries(args, kwargs)

        # Store in cache
        if self.config.cache:
            self._cache[cache_key] = result

        # Save checkpoint
        if self.config.checkpoint:
            self._save_checkpoint(result, cache_key)

        return result

    def _execute_with_retries(self, args: tuple, kwargs: dict) -> Any:
        """Execute function with retry logic."""
        import time

        last_exception = None

        for attempt in range(self.config.retries + 1):
            try:
                console.print(f"[blue]Running {self.name}...[/blue]")
                result = self.func(*args, **kwargs)
                console.print(f"[green]Completed {self.name}[/green]")
                return result

            except Exception as e:
                last_exception = e
                if attempt < self.config.retries:
                    console.print(
                        f"[yellow]Retry {attempt + 1}/{self.config.retries} "
                        f"for {self.name}: {e}[/yellow]"
                    )
                    time.sleep(self.config.retry_delay * (2 ** attempt))

        raise last_exception  # type: ignore

    def _compute_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Compute a cache key from function arguments."""
        # Simple hash-based key
        key_data = {
            "name": self.name,
            "args": str(args),
            "kwargs": str(sorted(kwargs.items())),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _get_checkpoint_path(self, cache_key: str) -> Path:
        """Get the checkpoint file path for a cache key."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.config.checkpoint_dir / f"{self.name}_{cache_key}.pkl"

    def _save_checkpoint(self, result: Any, cache_key: str) -> None:
        """Save result to checkpoint file."""
        checkpoint_path = self._get_checkpoint_path(cache_key)

        checkpoint_data = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "task_name": self.name,
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        console.print(f"[dim]Saved checkpoint for {self.name}[/dim]")

    def _load_checkpoint(self, checkpoint_path: Path) -> Any:
        """Load result from checkpoint file."""
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        return checkpoint_data["result"]

    def clear_cache(self) -> None:
        """Clear the in-memory cache for this task."""
        self._cache.clear()

    def clear_checkpoints(self) -> None:
        """Remove all checkpoint files for this task."""
        for checkpoint_file in self.config.checkpoint_dir.glob(f"{self.name}_*.pkl"):
            checkpoint_file.unlink()

    def __repr__(self) -> str:
        return f"Task({self.name!r})"


def task(
    cache: bool = False,
    checkpoint: bool = False,
    checkpoint_dir: str = ".geopipe/checkpoints",
    resources: dict[str, Any] | None = None,
    retries: int = 0,
    retry_delay: float = 1.0,
) -> Callable[[F], Task]:
    """
    Decorator to create a pipeline task.

    Args:
        cache: Enable in-memory caching of results
        checkpoint: Enable disk-based checkpointing
        checkpoint_dir: Directory for checkpoint files
        resources: Resource requirements (e.g., {"memory": "16GB"})
        retries: Number of retry attempts on failure
        retry_delay: Base delay between retries (exponential backoff)

    Returns:
        Decorator that wraps function in a Task

    Example:
        >>> @task(cache=True, checkpoint=True, resources={"memory": "8GB"})
        ... def extract_features(image_path):
        ...     # Process satellite imagery
        ...     return features
    """

    def decorator(func: F) -> Task:
        config = TaskConfig(
            cache=cache,
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            resources=resources,
            retries=retries,
            retry_delay=retry_delay,
        )
        return Task(func, config)

    return decorator
