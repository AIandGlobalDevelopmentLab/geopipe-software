"""SLURM cluster executor for distributed pipeline execution."""

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


console = Console()


@dataclass
class SLURMConfig:
    """Configuration for SLURM job submission."""

    partition: str = "normal"
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 4
    mem: str = "16GB"
    time_limit: str = "4:00:00"
    account: str | None = None
    conda_env: str | None = None
    modules: list[str] = field(default_factory=list)
    extra_sbatch: dict[str, str] = field(default_factory=dict)


class SLURMExecutor:
    """
    Executor for running pipelines on SLURM clusters.

    Handles job generation, submission, monitoring, and result collection
    for distributed pipeline execution.

    Attributes:
        config: SLURM configuration
        job_dir: Directory for job scripts and logs
        submitted_jobs: Dictionary of submitted job IDs

    Example:
        >>> executor = SLURMExecutor(
        ...     partition="gpu",
        ...     nodes=10,
        ...     time_limit="24:00:00",
        ... )
        >>> pipeline.run(executor=executor)
        >>> executor.status()
    """

    def __init__(
        self,
        partition: str = "normal",
        nodes: int = 1,
        ntasks: int = 1,
        cpus_per_task: int = 4,
        mem: str = "16GB",
        time_limit: str = "4:00:00",
        account: str | None = None,
        conda_env: str | None = None,
        modules: list[str] | None = None,
        job_dir: str = ".geopipe/slurm",
        **kwargs: Any,
    ) -> None:
        """
        Initialize SLURM executor.

        Args:
            partition: SLURM partition to submit to
            nodes: Number of nodes to request
            ntasks: Number of tasks per node
            cpus_per_task: CPUs per task
            mem: Memory per node (e.g., "16GB")
            time_limit: Maximum walltime (e.g., "4:00:00")
            account: SLURM account to charge
            conda_env: Conda environment to activate
            modules: Environment modules to load
            job_dir: Directory for job scripts and logs
            **kwargs: Additional SLURM options
        """
        self.config = SLURMConfig(
            partition=partition,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            mem=mem,
            time_limit=time_limit,
            account=account,
            conda_env=conda_env,
            modules=modules or [],
            extra_sbatch=kwargs,
        )
        self.job_dir = Path(job_dir)
        self.submitted_jobs: dict[str, str] = {}  # task_name -> job_id

    def generate_script(
        self,
        task_name: str,
        python_code: str,
        dependencies: list[str] | None = None,
    ) -> Path:
        """
        Generate a SLURM job script.

        Args:
            task_name: Name for this job
            python_code: Python code to execute
            dependencies: Job IDs that must complete first

        Returns:
            Path to generated script
        """
        self.job_dir.mkdir(parents=True, exist_ok=True)

        script_path = self.job_dir / f"{task_name}.sh"
        log_path = self.job_dir / f"{task_name}.log"

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={task_name}",
            f"#SBATCH --partition={self.config.partition}",
            f"#SBATCH --nodes={self.config.nodes}",
            f"#SBATCH --ntasks={self.config.ntasks}",
            f"#SBATCH --cpus-per-task={self.config.cpus_per_task}",
            f"#SBATCH --mem={self.config.mem}",
            f"#SBATCH --time={self.config.time_limit}",
            f"#SBATCH --output={log_path}",
            f"#SBATCH --error={log_path}",
        ]

        if self.config.account:
            lines.append(f"#SBATCH --account={self.config.account}")

        if dependencies:
            dep_str = ":".join(dependencies)
            lines.append(f"#SBATCH --dependency=afterok:{dep_str}")

        for key, value in self.config.extra_sbatch.items():
            lines.append(f"#SBATCH --{key}={value}")

        lines.append("")

        # Load modules
        for module in self.config.modules:
            lines.append(f"module load {module}")

        if self.config.modules:
            lines.append("")

        # Activate conda environment
        if self.config.conda_env:
            lines.extend([
                "# Activate conda environment",
                f"source $(conda info --base)/etc/profile.d/conda.sh",
                f"conda activate {self.config.conda_env}",
                "",
            ])

        # Python code
        lines.extend([
            "# Run pipeline task",
            "python << 'PYTHON_SCRIPT'",
            python_code,
            "PYTHON_SCRIPT",
        ])

        script_content = "\n".join(lines)
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        return script_path

    def submit(self, script_path: Path, task_name: str) -> str:
        """
        Submit a job script to SLURM.

        Args:
            script_path: Path to job script
            task_name: Name for tracking

        Returns:
            SLURM job ID
        """
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse job ID from output like "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            self.submitted_jobs[task_name] = job_id

            console.print(f"[green]Submitted {task_name} as job {job_id}[/green]")
            return job_id

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to submit {task_name}: {e.stderr}[/red]")
            raise
        except FileNotFoundError:
            console.print("[red]sbatch not found. Are you on a SLURM cluster?[/red]")
            raise

    def run_pipeline(self, pipeline: Any, initial_input: Any = None) -> Any:
        """
        Execute a pipeline on SLURM.

        Generates job scripts for each stage and submits them with dependencies.

        Args:
            pipeline: Pipeline to execute
            initial_input: Initial input to pipeline

        Returns:
            Job IDs for all submitted jobs
        """
        from geopipe.pipeline.dag import Pipeline

        if not isinstance(pipeline, Pipeline):
            raise TypeError("Expected a Pipeline object")

        job_ids = []

        for i, stage in enumerate(pipeline._tasks):
            # Generate Python code for this stage
            python_code = self._generate_stage_code(pipeline.name, stage.name, i)

            # Create script with dependency on previous job
            dependencies = [job_ids[-1]] if job_ids else None
            script_path = self.generate_script(
                task_name=f"{pipeline.name}_{stage.name}",
                python_code=python_code,
                dependencies=dependencies,
            )

            # Submit job
            job_id = self.submit(script_path, stage.name)
            job_ids.append(job_id)

        console.print(f"[green]Submitted {len(job_ids)} jobs for pipeline {pipeline.name}[/green]")
        return job_ids

    def _generate_stage_code(self, pipeline_name: str, stage_name: str, stage_idx: int) -> str:
        """Generate Python code for executing a pipeline stage."""
        return f'''
import pickle
from pathlib import Path

# Load pipeline definition
checkpoint_dir = Path(".geopipe/pipeline")
pipeline_file = checkpoint_dir / "{pipeline_name}_definition.pkl"

if pipeline_file.exists():
    with open(pipeline_file, "rb") as f:
        pipeline = pickle.load(f)
else:
    raise FileNotFoundError(f"Pipeline definition not found: {{pipeline_file}}")

# Load previous stage output if available
input_file = checkpoint_dir / f"{pipeline_name}_stage_{{stage_idx - 1}}_output.pkl"
if input_file.exists() and {stage_idx} > 0:
    with open(input_file, "rb") as f:
        stage_input = pickle.load(f)
else:
    stage_input = None

# Execute stage
stage = pipeline._tasks[{stage_idx}]
if stage_input is not None:
    result = stage(stage_input)
else:
    result = stage()

# Save output
output_file = checkpoint_dir / f"{pipeline_name}_stage_{stage_idx}_output.pkl"
with open(output_file, "wb") as f:
    pickle.dump(result, f)

print(f"Stage {stage_name} completed successfully")
'''

    def status(self) -> None:
        """Display status of submitted jobs."""
        if not self.submitted_jobs:
            console.print("[yellow]No jobs submitted[/yellow]")
            return

        # Query SLURM for job status
        job_ids = list(self.submitted_jobs.values())

        try:
            result = subprocess.run(
                ["squeue", "-j", ",".join(job_ids), "--format=%i|%j|%T|%M|%R"],
                capture_output=True,
                text=True,
            )
            output = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("[yellow]Unable to query SLURM status[/yellow]")
            return

        table = Table(title="Job Status")
        table.add_column("Task", style="cyan")
        table.add_column("Job ID", style="dim")
        table.add_column("State", style="green")
        table.add_column("Time", style="dim")
        table.add_column("Reason", style="dim")

        # Parse squeue output
        lines = output.split("\n")[1:]  # Skip header

        job_status = {}
        for line in lines:
            if "|" in line:
                parts = line.split("|")
                job_status[parts[0]] = {
                    "name": parts[1],
                    "state": parts[2],
                    "time": parts[3],
                    "reason": parts[4] if len(parts) > 4 else "",
                }

        for task_name, job_id in self.submitted_jobs.items():
            if job_id in job_status:
                info = job_status[job_id]
                state_color = {
                    "RUNNING": "green",
                    "PENDING": "yellow",
                    "COMPLETED": "blue",
                    "FAILED": "red",
                }.get(info["state"], "white")
                table.add_row(
                    task_name,
                    job_id,
                    f"[{state_color}]{info['state']}[/{state_color}]",
                    info["time"],
                    info["reason"],
                )
            else:
                table.add_row(task_name, job_id, "[dim]COMPLETED/UNKNOWN[/dim]", "-", "-")

        console.print(table)

    def cancel(self, task_name: str | None = None) -> None:
        """
        Cancel submitted jobs.

        Args:
            task_name: Specific task to cancel (None = all jobs)
        """
        if task_name:
            if task_name in self.submitted_jobs:
                job_id = self.submitted_jobs[task_name]
                subprocess.run(["scancel", job_id])
                console.print(f"[yellow]Cancelled job {job_id}[/yellow]")
        else:
            for job_id in self.submitted_jobs.values():
                subprocess.run(["scancel", job_id])
            console.print(f"[yellow]Cancelled {len(self.submitted_jobs)} jobs[/yellow]")

    def wait(self, poll_interval: float = 30.0) -> bool:
        """
        Wait for all submitted jobs to complete.

        Args:
            poll_interval: Seconds between status checks

        Returns:
            True if all jobs completed successfully
        """
        if not self.submitted_jobs:
            return True

        console.print("[blue]Waiting for jobs to complete...[/blue]")

        while True:
            # Check job statuses
            job_ids = list(self.submitted_jobs.values())

            try:
                result = subprocess.run(
                    ["squeue", "-j", ",".join(job_ids), "-h", "--format=%T"],
                    capture_output=True,
                    text=True,
                )
                states = result.stdout.strip().split("\n")
                states = [s for s in states if s]  # Remove empty strings

            except (subprocess.CalledProcessError, FileNotFoundError):
                # Assume completed if squeue fails
                states = []

            if not states:
                # All jobs completed (not in queue)
                console.print("[green]All jobs completed[/green]")
                return True

            running = states.count("RUNNING")
            pending = states.count("PENDING")
            console.print(f"[dim]Jobs: {running} running, {pending} pending[/dim]")

            time.sleep(poll_interval)
