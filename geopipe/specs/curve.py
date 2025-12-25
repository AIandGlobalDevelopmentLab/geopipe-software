"""Specification curve analysis for robustness checking."""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from geopipe.specs.variants import Spec


console = Console()


class SpecificationCurve:
    """
    Analyze and visualize results across robustness specifications.

    Provides tools for comparing estimates across many specifications,
    ranking dimensions by influence, and generating publication-ready outputs.

    Attributes:
        specs: List of specifications that were run
        results_pattern: Glob pattern for result files
        estimate_col: Column name for point estimates
        se_col: Column name for standard errors
        results_dir: Base directory for results

    Example:
        >>> curve = SpecificationCurve(
        ...     specs=dsl.expand(),
        ...     results_pattern="results/{spec_name}/estimates.csv",
        ...     estimate_col="treatment_effect",
        ... )
        >>> summary = curve.compute_summary()
        >>> curve.plot("figures/spec_curve.pdf")
    """

    def __init__(
        self,
        specs: list[Spec],
        results_pattern: str,
        estimate_col: str = "estimate",
        se_col: str = "std_error",
        results_dir: str = ".",
    ) -> None:
        """
        Initialize specification curve analysis.

        Args:
            specs: List of Spec objects
            results_pattern: Pattern with {spec_name} for finding results
            estimate_col: Column containing point estimates
            se_col: Column containing standard errors
            results_dir: Base directory for results
        """
        self.specs = specs
        self.results_pattern = results_pattern
        self.estimate_col = estimate_col
        self.se_col = se_col
        self.results_dir = Path(results_dir)
        self._results: pd.DataFrame | None = None

    def load_results(self) -> pd.DataFrame:
        """
        Load all results into a single DataFrame.

        Returns:
            DataFrame with spec parameters, estimates, and standard errors
        """
        records = []

        for spec in self.specs:
            # Build file path from pattern
            file_pattern = self.results_pattern.replace("{spec_name}", spec.name)
            file_pattern = file_pattern.replace("*", spec.name)

            full_pattern = str(self.results_dir / file_pattern)
            matches = glob(full_pattern)

            if not matches:
                continue

            # Load first match
            file_path = Path(matches[0])

            try:
                if file_path.suffix == ".csv":
                    df = pd.read_csv(file_path)
                elif file_path.suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                else:
                    continue

                # Extract estimate and SE
                if self.estimate_col in df.columns:
                    estimate = df[self.estimate_col].mean()
                    se = df[self.se_col].mean() if self.se_col in df.columns else None

                    record = {
                        "spec_name": spec.name,
                        "estimate": estimate,
                        "std_error": se,
                        **spec.params,
                    }
                    records.append(record)

            except Exception:
                continue

        self._results = pd.DataFrame(records)
        return self._results

    def compute_summary(self) -> dict[str, Any]:
        """
        Compute summary statistics across specifications.

        Returns:
            Dictionary with summary statistics
        """
        if self._results is None:
            self.load_results()

        if self._results is None or self._results.empty:
            return {"error": "No results loaded"}

        estimates = self._results["estimate"]

        summary = {
            "n_specs": len(self._results),
            "mean_estimate": estimates.mean(),
            "median_estimate": estimates.median(),
            "std_estimate": estimates.std(),
            "min_estimate": estimates.min(),
            "max_estimate": estimates.max(),
            "pct_positive": (estimates > 0).mean() * 100,
        }

        # Add significance stats if SE available
        if "std_error" in self._results.columns and self._results["std_error"].notna().any():
            # Two-tailed test at 5%
            z_stat = abs(self._results["estimate"] / self._results["std_error"])
            summary["pct_significant_05"] = (z_stat > 1.96).mean() * 100
            summary["pct_significant_10"] = (z_stat > 1.645).mean() * 100

        return summary

    def rank_by_influence(self) -> pd.DataFrame:
        """
        Rank dimensions by their influence on estimates.

        Computes the variance in estimates explained by each dimension.

        Returns:
            DataFrame with dimensions ranked by influence
        """
        if self._results is None:
            self.load_results()

        if self._results is None or self._results.empty:
            return pd.DataFrame()

        # Get dimension columns (exclude spec_name, estimate, std_error)
        exclude_cols = {"spec_name", "estimate", "std_error"}
        dim_cols = [c for c in self._results.columns if c not in exclude_cols]

        if not dim_cols:
            return pd.DataFrame()

        influence_records = []

        total_variance = self._results["estimate"].var()
        if total_variance == 0:
            return pd.DataFrame()

        for dim in dim_cols:
            # Compute between-group variance
            group_means = self._results.groupby(dim)["estimate"].mean()
            between_variance = group_means.var()

            # Compute range of estimates by this dimension
            estimate_range = group_means.max() - group_means.min()

            influence_records.append(
                {
                    "dimension": dim,
                    "variance_ratio": between_variance / total_variance if total_variance > 0 else 0,
                    "estimate_range": estimate_range,
                    "n_values": self._results[dim].nunique(),
                }
            )

        influence_df = pd.DataFrame(influence_records)
        influence_df = influence_df.sort_values("variance_ratio", ascending=False)

        return influence_df

    def to_latex(self, caption: str = "Specification Curve Summary") -> str:
        """
        Generate LaTeX table summarizing specifications.

        Args:
            caption: Table caption

        Returns:
            LaTeX table string
        """
        if self._results is None:
            self.load_results()

        if self._results is None or self._results.empty:
            return ""

        summary = self.compute_summary()

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            r"\begin{tabular}{lr}",
            r"\hline",
            r"Statistic & Value \\",
            r"\hline",
            f"Number of specifications & {summary.get('n_specs', 0)} \\\\",
            f"Mean estimate & {summary.get('mean_estimate', 0):.4f} \\\\",
            f"Median estimate & {summary.get('median_estimate', 0):.4f} \\\\",
            f"Std. deviation & {summary.get('std_estimate', 0):.4f} \\\\",
            f"Range & [{summary.get('min_estimate', 0):.4f}, {summary.get('max_estimate', 0):.4f}] \\\\",
            f"\\% Positive & {summary.get('pct_positive', 0):.1f}\\% \\\\",
        ]

        if "pct_significant_05" in summary:
            lines.append(f"\\% Significant (p<0.05) & {summary['pct_significant_05']:.1f}\\% \\\\")

        lines.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """
        Generate markdown summary of specification curve.

        Returns:
            Markdown string
        """
        if self._results is None:
            self.load_results()

        if self._results is None or self._results.empty:
            return "No results loaded."

        summary = self.compute_summary()

        lines = [
            "# Specification Curve Summary",
            "",
            f"**Specifications analyzed:** {summary.get('n_specs', 0)}",
            "",
            "## Estimate Distribution",
            "",
            f"- Mean: {summary.get('mean_estimate', 0):.4f}",
            f"- Median: {summary.get('median_estimate', 0):.4f}",
            f"- Std. Dev: {summary.get('std_estimate', 0):.4f}",
            f"- Range: [{summary.get('min_estimate', 0):.4f}, {summary.get('max_estimate', 0):.4f}]",
            f"- % Positive: {summary.get('pct_positive', 0):.1f}%",
        ]

        if "pct_significant_05" in summary:
            lines.extend(
                [
                    "",
                    "## Significance",
                    "",
                    f"- % Significant (p<0.05): {summary['pct_significant_05']:.1f}%",
                    f"- % Significant (p<0.10): {summary['pct_significant_10']:.1f}%",
                ]
            )

        # Add influence ranking
        influence = self.rank_by_influence()
        if not influence.empty:
            lines.extend(
                [
                    "",
                    "## Dimension Influence",
                    "",
                    "| Dimension | Variance Ratio | Estimate Range |",
                    "|-----------|----------------|----------------|",
                ]
            )
            for _, row in influence.iterrows():
                lines.append(
                    f"| {row['dimension']} | {row['variance_ratio']:.3f} | {row['estimate_range']:.4f} |"
                )

        return "\n".join(lines)

    def plot(self, output_path: Path | str | None = None) -> Any:
        """
        Generate specification curve plot.

        Args:
            output_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            console.print("[yellow]matplotlib not installed - cannot generate plot[/yellow]")
            return None

        if self._results is None:
            self.load_results()

        if self._results is None or self._results.empty:
            console.print("[yellow]No results to plot[/yellow]")
            return None

        # Sort by estimate for specification curve
        sorted_results = self._results.sort_values("estimate").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot estimates
        x = range(len(sorted_results))
        estimates = sorted_results["estimate"]

        ax.scatter(x, estimates, s=20, alpha=0.7, color="steelblue")

        # Add confidence intervals if SE available
        if "std_error" in sorted_results.columns:
            se = sorted_results["std_error"]
            ax.fill_between(
                x,
                estimates - 1.96 * se,
                estimates + 1.96 * se,
                alpha=0.2,
                color="steelblue",
            )

        # Add reference line at zero
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

        # Add median line
        ax.axhline(y=estimates.median(), color="green", linestyle="-", alpha=0.5, label="Median")

        ax.set_xlabel("Specification (sorted by estimate)")
        ax.set_ylabel("Estimate")
        ax.set_title("Specification Curve")
        ax.legend()

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            console.print(f"[green]Saved plot to {output_path}[/green]")

        return fig

    def display(self) -> None:
        """Display specification curve summary in terminal."""
        if self._results is None:
            self.load_results()

        if self._results is None or self._results.empty:
            console.print("[yellow]No results loaded[/yellow]")
            return

        summary = self.compute_summary()

        # Summary table
        table = Table(title="Specification Curve Summary")
        table.add_column("Statistic", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Specifications", str(summary.get("n_specs", 0)))
        table.add_row("Mean estimate", f"{summary.get('mean_estimate', 0):.4f}")
        table.add_row("Median estimate", f"{summary.get('median_estimate', 0):.4f}")
        table.add_row("Std. deviation", f"{summary.get('std_estimate', 0):.4f}")
        table.add_row("% Positive", f"{summary.get('pct_positive', 0):.1f}%")

        if "pct_significant_05" in summary:
            table.add_row("% Significant (p<0.05)", f"{summary['pct_significant_05']:.1f}%")

        console.print(table)

        # Influence table
        influence = self.rank_by_influence()
        if not influence.empty:
            influence_table = Table(title="Dimension Influence")
            influence_table.add_column("Dimension", style="cyan")
            influence_table.add_column("Variance Ratio", justify="right")
            influence_table.add_column("Estimate Range", justify="right")

            for _, row in influence.iterrows():
                influence_table.add_row(
                    str(row["dimension"]),
                    f"{row['variance_ratio']:.3f}",
                    f"{row['estimate_range']:.4f}",
                )

            console.print(influence_table)

    def __repr__(self) -> str:
        n_loaded = len(self._results) if self._results is not None else 0
        return f"SpecificationCurve(specs={len(self.specs)}, loaded={n_loaded})"
