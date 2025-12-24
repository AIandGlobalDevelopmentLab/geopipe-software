"""Robustness specification management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from rich.console import Console
from rich.table import Table


console = Console()


@dataclass
class Spec:
    """
    A single analysis specification for robustness checking.

    Represents one configuration of analysis parameters that can be
    compared against other specifications.

    Attributes:
        name: Unique identifier (e.g., "MAIN", "ROBUST_BUFFER")
        params: Dictionary of specification parameters
        description: Human-readable description

    Example:
        >>> spec = Spec(
        ...     "ROBUST_BUFFER",
        ...     buffer_km=10,
        ...     include_ntl=True,
        ...     description="Larger buffer for spatial matching",
        ... )
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __init__(self, name: str, description: str = "", **kwargs: Any) -> None:
        self.name = name
        self.description = description
        self.params = kwargs

    def __getattr__(self, key: str) -> Any:
        """Allow attribute-style access to params."""
        if key in {"name", "description", "params"}:
            return object.__getattribute__(self, key)
        params = object.__getattribute__(self, "params")
        if key in params:
            return params[key]
        raise AttributeError(f"Spec has no parameter '{key}'")

    def __repr__(self) -> str:
        return f"Spec({self.name!r}, {self.params})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            **self.params,
        }


class SpecRegistry:
    """
    Registry for managing multiple analysis specifications.

    Provides iteration over specs, parallel execution support,
    and result comparison utilities.

    Attributes:
        specs: List of Spec objects
        results_dir: Directory for specification results

    Example:
        >>> registry = SpecRegistry([
        ...     Spec("MAIN", buffer_km=5, include_ntl=True),
        ...     Spec("ROBUST_BUFFER", buffer_km=10, include_ntl=True),
        ...     Spec("ROBUST_NO_NTL", buffer_km=5, include_ntl=False),
        ... ])
        >>> for spec in registry:
        ...     pipeline.run(spec=spec)
        >>> registry.compare_results("results/*/estimates.csv")
    """

    def __init__(
        self,
        specs: list[Spec] | None = None,
        results_dir: str = "results",
    ) -> None:
        """
        Initialize spec registry.

        Args:
            specs: List of Spec objects
            results_dir: Base directory for results
        """
        self.specs = specs or []
        self.results_dir = Path(results_dir)
        self._results: dict[str, pd.DataFrame] = {}

    def add(self, spec: Spec) -> "SpecRegistry":
        """
        Add a specification to the registry.

        Args:
            spec: Spec to add

        Returns:
            Self for chaining
        """
        self.specs.append(spec)
        return self

    def get(self, name: str) -> Spec | None:
        """Get a spec by name."""
        for spec in self.specs:
            if spec.name == name:
                return spec
        return None

    def __iter__(self) -> Iterator[Spec]:
        """Iterate over specifications."""
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, key: str | int) -> Spec:
        if isinstance(key, int):
            return self.specs[key]
        spec = self.get(key)
        if spec is None:
            raise KeyError(f"No spec named '{key}'")
        return spec

    def load_results(self, pattern: str) -> dict[str, pd.DataFrame]:
        """
        Load results from files matching a pattern.

        Pattern should use {spec} or * for spec name substitution.

        Args:
            pattern: Glob pattern for result files

        Returns:
            Dictionary mapping spec names to DataFrames
        """
        from glob import glob

        results = {}

        for spec in self.specs:
            spec_pattern = pattern.replace("{spec}", spec.name)
            spec_pattern = spec_pattern.replace("*", spec.name)

            matches = glob(str(self.results_dir / spec_pattern))

            if matches:
                # Load first match
                file_path = Path(matches[0])
                if file_path.suffix == ".csv":
                    results[spec.name] = pd.read_csv(file_path)
                elif file_path.suffix == ".parquet":
                    results[spec.name] = pd.read_parquet(file_path)
                else:
                    results[spec.name] = pd.read_csv(file_path)

        self._results = results
        return results

    def compare_results(
        self,
        pattern: str,
        estimate_col: str = "estimate",
        se_col: str = "std_error",
    ) -> pd.DataFrame:
        """
        Compare results across specifications.

        Args:
            pattern: Glob pattern for result files
            estimate_col: Column name for point estimates
            se_col: Column name for standard errors

        Returns:
            DataFrame comparing estimates across specs
        """
        results = self.load_results(pattern)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return pd.DataFrame()

        comparison_data = []

        for spec_name, df in results.items():
            if estimate_col in df.columns:
                row = {"spec": spec_name, "estimate": df[estimate_col].mean()}

                if se_col in df.columns:
                    row["std_error"] = df[se_col].mean()

                comparison_data.append(row)

        comparison = pd.DataFrame(comparison_data)

        # Display comparison
        self._display_comparison(comparison)

        return comparison

    def _display_comparison(self, comparison: pd.DataFrame) -> None:
        """Display comparison table."""
        table = Table(title="Specification Comparison")
        table.add_column("Specification", style="cyan")
        table.add_column("Estimate", justify="right", style="green")

        if "std_error" in comparison.columns:
            table.add_column("Std. Error", justify="right", style="dim")

        for _, row in comparison.iterrows():
            if "std_error" in comparison.columns:
                table.add_row(
                    row["spec"],
                    f"{row['estimate']:.4f}",
                    f"{row['std_error']:.4f}",
                )
            else:
                table.add_row(row["spec"], f"{row['estimate']:.4f}")

        console.print(table)

    def to_latex(
        self,
        pattern: str,
        estimate_col: str = "estimate",
        se_col: str = "std_error",
        caption: str = "Robustness Specifications",
    ) -> str:
        """
        Generate LaTeX table comparing specifications.

        Args:
            pattern: Glob pattern for result files
            estimate_col: Column name for point estimates
            se_col: Column name for standard errors
            caption: Table caption

        Returns:
            LaTeX table string
        """
        comparison = self.compare_results(pattern, estimate_col, se_col)

        if comparison.empty:
            return ""

        # Generate LaTeX
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            r"\begin{tabular}{lcc}",
            r"\hline",
            r"Specification & Estimate & Std. Error \\",
            r"\hline",
        ]

        for _, row in comparison.iterrows():
            est = f"{row['estimate']:.4f}"
            se = f"{row.get('std_error', 0):.4f}"
            lines.append(f"{row['spec']} & {est} & ({se}) \\\\")

        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def summary(self) -> str:
        """Return a summary of all specifications."""
        lines = [f"SpecRegistry ({len(self.specs)} specifications):", ""]

        for spec in self.specs:
            lines.append(f"  {spec.name}:")
            if spec.description:
                lines.append(f"    Description: {spec.description}")
            for key, value in spec.params.items():
                lines.append(f"    {key}: {value}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        names = [s.name for s in self.specs]
        return f"SpecRegistry({names})"
