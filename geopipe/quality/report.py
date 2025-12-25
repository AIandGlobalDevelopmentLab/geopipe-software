"""Quality report generation for geopipe fusion schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from geopipe.quality.checks import (
    CrossSourceCheck,
    IssueSeverity,
    QualityCheck,
    QualityIssue,
    DEFAULT_CHECKS,
)

if TYPE_CHECKING:
    from geopipe.fusion.schema import FusionSchema


console = Console()


class QualityReport:
    """
    Aggregated quality report for a fusion schema.

    Provides quality auditing, scoring, and export functionality
    for data fusion workflows.

    Attributes:
        schema: The fusion schema being audited
        issues: List of detected quality issues
        source_scores: Quality scores per source (0-100)
        overall_score: Overall quality score (0-100)
        timestamp: When the audit was run

    Example:
        >>> schema = FusionSchema.from_yaml("schema.yaml")
        >>> report = QualityReport.from_schema(schema)
        >>> print(report.summary())
        >>> report.to_markdown("quality_report.md")
    """

    def __init__(self, schema: "FusionSchema") -> None:
        """Initialize empty quality report."""
        self.schema = schema
        self.issues: list[QualityIssue] = []
        self.source_scores: dict[str, float] = {}
        self.overall_score: float = 100.0
        self.timestamp: datetime = datetime.now()
        self._checks_run: list[str] = []

    @classmethod
    def from_schema(
        cls,
        schema: "FusionSchema",
        checks: list[QualityCheck] | None = None,
        sample_size: int = 1000,
    ) -> "QualityReport":
        """
        Run quality audit on a fusion schema.

        Args:
            schema: Fusion schema to audit
            checks: Quality checks to run (uses DEFAULT_CHECKS if None)
            sample_size: Sample size for data-level checks

        Returns:
            QualityReport with all detected issues
        """
        report = cls(schema)
        checks = checks or DEFAULT_CHECKS

        # Run cross-source checks first
        cross_checks = [c for c in checks if isinstance(c, CrossSourceCheck)]
        single_checks = [c for c in checks if not isinstance(c, CrossSourceCheck)]

        for check in cross_checks:
            try:
                issues = check.check_all(schema.sources, schema)
                report.issues.extend(issues)
                report._checks_run.append(check.name)
            except Exception as e:
                # Log but don't fail on check errors
                report.issues.append(
                    QualityIssue(
                        source_name="(system)",
                        category=check.category,
                        severity=IssueSeverity.INFO,
                        message=f"Check '{check.name}' failed: {e}",
                    )
                )

        # Run per-source checks
        for source in schema.sources:
            for check in single_checks:
                try:
                    issues = check.check(source, schema, sample_size)
                    report.issues.extend(issues)
                    if check.name not in report._checks_run:
                        report._checks_run.append(check.name)
                except Exception as e:
                    report.issues.append(
                        QualityIssue(
                            source_name=source.name,
                            category=check.category,
                            severity=IssueSeverity.INFO,
                            message=f"Check '{check.name}' failed: {e}",
                        )
                    )

        # Compute scores
        report.compute_scores()

        return report

    def add_issue(self, issue: QualityIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)

    def compute_scores(self) -> None:
        """Compute quality scores from issues."""
        # Score weights by severity
        weights = {
            IssueSeverity.ERROR: 25,
            IssueSeverity.WARNING: 10,
            IssueSeverity.INFO: 2,
        }

        # Compute per-source scores
        source_issues: dict[str, list[QualityIssue]] = {}
        cross_source_issues: list[QualityIssue] = []

        for issue in self.issues:
            if "," in issue.source_name or issue.source_name == "(cross-source)":
                cross_source_issues.append(issue)
            else:
                if issue.source_name not in source_issues:
                    source_issues[issue.source_name] = []
                source_issues[issue.source_name].append(issue)

        # Calculate per-source scores
        for source_name, issues in source_issues.items():
            penalty = sum(weights.get(i.severity, 0) for i in issues)
            self.source_scores[source_name] = max(0, 100 - penalty)

        # Add sources with no issues
        for source in self.schema.sources:
            if source.name not in self.source_scores:
                self.source_scores[source.name] = 100

        # Calculate overall score
        if self.source_scores:
            avg_source_score = sum(self.source_scores.values()) / len(self.source_scores)
        else:
            avg_source_score = 100

        # Penalize for cross-source issues
        cross_penalty = sum(weights.get(i.severity, 0) for i in cross_source_issues)
        self.overall_score = max(0, avg_source_score - cross_penalty)

    @property
    def has_errors(self) -> bool:
        """Check if report contains any errors."""
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if report contains any warnings."""
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    @property
    def fixable_issues(self) -> list[QualityIssue]:
        """Get list of auto-fixable issues."""
        return [i for i in self.issues if i.auto_fixable]

    def filter(
        self,
        severity: IssueSeverity | None = None,
        source: str | None = None,
        category: str | None = None,
    ) -> list[QualityIssue]:
        """Filter issues by criteria."""
        result = self.issues

        if severity:
            result = [i for i in result if i.severity == severity]

        if source:
            result = [i for i in result if source in i.source_name]

        if category:
            result = [i for i in result if i.category.value == category]

        return result

    def summary(self) -> str:
        """Generate rich-formatted summary for terminal display."""
        lines = []

        # Header
        score_color = "green" if self.overall_score >= 80 else "yellow" if self.overall_score >= 60 else "red"

        header = Text()
        header.append("DATA QUALITY AUDIT REPORT\n", style="bold")
        header.append(f"Schema: {self.schema.name}\n")
        header.append(f"Sources: {len(self.schema.sources)}\n")
        header.append(f"Checks run: {len(self._checks_run)}\n")
        header.append(f"Overall Score: ", style="dim")
        header.append(f"{self.overall_score:.0f}/100", style=f"bold {score_color}")

        lines.append(Panel(header, title="Quality Audit", border_style="blue"))

        # Per-source scores
        if self.source_scores:
            table = Table(title="Source Quality Scores", show_header=True)
            table.add_column("Source", style="cyan")
            table.add_column("Score", justify="right")
            table.add_column("Issues", justify="right")

            for source_name, score in sorted(self.source_scores.items()):
                source_issues = len([i for i in self.issues if source_name in i.source_name])
                score_style = "green" if score >= 80 else "yellow" if score >= 60 else "red"
                table.add_row(source_name, f"[{score_style}]{score:.0f}[/]", str(source_issues))

            lines.append(table)

        # Issues summary
        if self.issues:
            issues_table = Table(title="Issues Found", show_header=True)
            issues_table.add_column("Severity", width=10)
            issues_table.add_column("Source", width=20)
            issues_table.add_column("Message")

            severity_styles = {
                IssueSeverity.ERROR: "[bold red]ERROR[/]",
                IssueSeverity.WARNING: "[yellow]WARNING[/]",
                IssueSeverity.INFO: "[dim]INFO[/]",
            }

            for issue in self.issues[:20]:  # Limit display
                issues_table.add_row(
                    severity_styles.get(issue.severity, str(issue.severity)),
                    issue.source_name[:20],
                    issue.message[:60] + ("..." if len(issue.message) > 60 else ""),
                )

            if len(self.issues) > 20:
                issues_table.add_row("...", f"+{len(self.issues) - 20} more", "")

            lines.append(issues_table)
        else:
            lines.append(Text("No issues found!", style="green bold"))

        # Render to string
        from io import StringIO

        buffer = StringIO()
        temp_console = Console(file=buffer, force_terminal=True, width=100)

        for item in lines:
            temp_console.print(item)

        return buffer.getvalue()

    def to_markdown(self, path: str | Path | None = None) -> str:
        """
        Generate markdown report.

        Args:
            path: If provided, write to file

        Returns:
            Markdown string
        """
        lines = [
            "# Data Quality Audit Report",
            "",
            f"**Schema:** {self.schema.name}",
            f"**Timestamp:** {self.timestamp.isoformat()}",
            f"**Overall Score:** {self.overall_score:.0f}/100",
            "",
            "## Source Quality Scores",
            "",
            "| Source | Score | Issues |",
            "|--------|-------|--------|",
        ]

        for source_name, score in sorted(self.source_scores.items()):
            source_issues = len([i for i in self.issues if source_name in i.source_name])
            lines.append(f"| {source_name} | {score:.0f} | {source_issues} |")

        lines.extend(["", "## Issues", ""])

        if self.issues:
            # Group by severity
            errors = [i for i in self.issues if i.severity == IssueSeverity.ERROR]
            warnings = [i for i in self.issues if i.severity == IssueSeverity.WARNING]
            infos = [i for i in self.issues if i.severity == IssueSeverity.INFO]

            if errors:
                lines.append("### Errors")
                lines.append("")
                for issue in errors:
                    lines.append(f"- **{issue.source_name}**: {issue.message}")
                lines.append("")

            if warnings:
                lines.append("### Warnings")
                lines.append("")
                for issue in warnings:
                    lines.append(f"- **{issue.source_name}**: {issue.message}")
                lines.append("")

            if infos:
                lines.append("### Info")
                lines.append("")
                for issue in infos:
                    lines.append(f"- **{issue.source_name}**: {issue.message}")
                lines.append("")
        else:
            lines.append("No issues found.")
            lines.append("")

        content = "\n".join(lines)

        if path:
            Path(path).write_text(content)

        return content

    def to_latex(self, caption: str = "Data Quality Summary") -> str:
        """
        Generate LaTeX table for supplementary materials.

        Args:
            caption: Table caption

        Returns:
            LaTeX table string
        """
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            r"\begin{tabular}{lrr}",
            r"\hline",
            r"Source & Score & Issues \\",
            r"\hline",
        ]

        for source_name, score in sorted(self.source_scores.items()):
            source_issues = len([i for i in self.issues if source_name in i.source_name])
            # Escape underscores for LaTeX
            safe_name = source_name.replace("_", r"\_")
            lines.append(f"{safe_name} & {score:.0f} & {source_issues} \\\\")

        lines.extend(
            [
                r"\hline",
                f"Overall & {self.overall_score:.0f} & {len(self.issues)} \\\\",
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export report as dictionary."""
        return {
            "schema_name": self.schema.name,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "source_scores": self.source_scores,
            "issues": [
                {
                    "source_name": i.source_name,
                    "category": i.category.value,
                    "severity": i.severity.value,
                    "message": i.message,
                    "details": i.details,
                    "auto_fixable": i.auto_fixable,
                }
                for i in self.issues
            ],
            "checks_run": self._checks_run,
        }

    def apply_fixes(
        self,
        interactive: bool = True,
    ) -> list[tuple[QualityIssue, bool]]:
        """
        Apply auto-fixes for fixable issues.

        Args:
            interactive: If True, prompt for each fix

        Returns:
            List of (issue, success) tuples
        """
        results: list[tuple[QualityIssue, bool]] = []

        for issue in self.fixable_issues:
            if interactive:
                console.print(f"\n[yellow]Fix available:[/yellow] {issue.message}")
                console.print(f"  Action: {issue.fix_description}")

                import click

                if not click.confirm("Apply fix?"):
                    results.append((issue, False))
                    continue

            # Apply fix based on issue type
            success = self._apply_fix(issue)
            results.append((issue, success))

            if success:
                console.print(f"  [green]Fixed![/green]")
            else:
                console.print(f"  [red]Fix failed[/red]")

        return results

    def _apply_fix(self, issue: QualityIssue) -> bool:
        """Apply a single fix. Override in subclass for custom fixes."""
        # Default implementation - no actual fixes applied
        # Subclasses or extensions can implement actual fix logic
        return False

    def __repr__(self) -> str:
        return (
            f"QualityReport(schema={self.schema.name!r}, "
            f"score={self.overall_score:.0f}, issues={len(self.issues)})"
        )
