"""Command-line interface for geopipe."""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table


console = Console()


@click.group()
@click.version_option(package_name="geopipe")
def main() -> None:
    """geopipe: Geospatial data pipeline framework for causal inference."""
    pass


@main.command()
@click.argument("schema_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Override output path from schema",
)
@click.option(
    "--dry-run", "-n",
    is_flag=True,
    help="Validate schema without executing",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show verbose output",
)
def fuse(
    schema_file: str,
    output: Optional[str],
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Execute a data fusion schema.

    SCHEMA_FILE is a YAML file defining the fusion sources and configuration.

    Example:
        geopipe fuse sources.yaml
        geopipe fuse sources.yaml --output data/output.parquet
        geopipe fuse sources.yaml --dry-run
    """
    from geopipe.fusion.schema import FusionSchema

    console.print(f"[blue]Loading schema from {schema_file}[/blue]")

    try:
        schema = FusionSchema.from_yaml(schema_file)
    except Exception as e:
        console.print(f"[red]Error loading schema: {e}[/red]")
        raise click.Abort()

    if verbose:
        console.print(schema.summary())

    # Override output if specified
    if output:
        schema.output = output

    # Validate sources
    issues = schema.validate_sources()
    if issues:
        console.print("[yellow]Validation issues:[/yellow]")
        for issue in issues:
            console.print(f"  - {issue}")

        if dry_run:
            raise click.Abort()

    if dry_run:
        console.print("[green]Schema is valid[/green]")
        return

    # Execute fusion
    try:
        result = schema.execute()
        console.print(f"[green]Fused {len(result)} records[/green]")
    except Exception as e:
        console.print(f"[red]Error during fusion: {e}[/red]")
        raise click.Abort()


@main.command()
@click.argument("schema_file", type=click.Path(exists=True))
def validate(schema_file: str) -> None:
    """
    Validate a fusion schema without executing.

    Checks that all data sources are accessible and the schema is well-formed.
    """
    from geopipe.fusion.schema import FusionSchema

    try:
        schema = FusionSchema.from_yaml(schema_file)
    except Exception as e:
        console.print(f"[red]Invalid schema: {e}[/red]")
        raise click.Abort()

    console.print(schema.summary())
    console.print()

    issues = schema.validate_sources()

    if issues:
        console.print("[yellow]Issues found:[/yellow]")
        for issue in issues:
            console.print(f"  [red]✗[/red] {issue}")
        raise click.Abort()
    else:
        console.print("[green]✓ All sources valid[/green]")


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="schema.yaml",
    help="Output path for generated schema",
)
@click.option(
    "--name", "-n",
    default="my_fusion",
    help="Name for the fusion schema",
)
def init(output: str, name: str) -> None:
    """
    Generate a template fusion schema YAML file.

    Creates a starting point for defining your data fusion configuration.
    """
    template = f'''# geopipe Fusion Schema
# Generated template - customize for your data sources

name: {name}
resolution: 5km
temporal_range:
  - "2010-01-01"
  - "2020-12-31"

sources:
  # Raster data (satellite imagery)
  - type: raster
    name: nightlights
    path: data/viirs/*.tif
    aggregation: mean

  # Tabular data with spatial join
  - type: tabular
    name: conflict_events
    path: data/acled.csv
    lat_col: latitude
    lon_col: longitude
    time_col: event_date
    spatial_join: buffer_10km
    temporal_align: yearly_sum

  # Another tabular source
  - type: tabular
    name: aid_projects
    path: data/aiddata.csv
    lat_col: lat
    lon_col: lon
    spatial_join: nearest

output: data/fused_output.parquet
target_crs: EPSG:4326
'''

    output_path = Path(output)

    if output_path.exists():
        if not click.confirm(f"{output} exists. Overwrite?"):
            raise click.Abort()

    output_path.write_text(template)
    console.print(f"[green]Created template schema at {output}[/green]")
    console.print("Edit the file to configure your data sources, then run:")
    console.print(f"  geopipe fuse {output}")


@main.command()
@click.argument("pipeline_dir", type=click.Path(), default=".geopipe")
def status(pipeline_dir: str) -> None:
    """
    Show status of pipeline checkpoints.

    Displays information about saved pipeline states and checkpoints.
    """
    import json

    pipeline_path = Path(pipeline_dir)

    if not pipeline_path.exists():
        console.print("[yellow]No pipeline checkpoints found[/yellow]")
        return

    # Find pipeline state files
    state_files = list(pipeline_path.glob("**/pipeline/*_state.json"))

    if not state_files:
        console.print("[yellow]No pipeline states found[/yellow]")
        return

    table = Table(title="Pipeline Status")
    table.add_column("Pipeline", style="cyan")
    table.add_column("Progress", style="green")
    table.add_column("Last Updated", style="dim")

    for state_file in state_files:
        with open(state_file) as f:
            state = json.load(f)

        name = state.get("name", "unknown")
        completed = state.get("last_completed_stage", 0) + 1
        total = state.get("total_stages", 0)
        timestamp = state.get("timestamp", "unknown")

        progress = f"{completed}/{total} stages"
        table.add_row(name, progress, timestamp)

    console.print(table)


@main.command()
@click.argument("pipeline_dir", type=click.Path(), default=".geopipe")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def clean(pipeline_dir: str, force: bool) -> None:
    """
    Remove all pipeline checkpoints.

    Clears saved states and checkpoint files to start fresh.
    """
    import shutil

    pipeline_path = Path(pipeline_dir)

    if not pipeline_path.exists():
        console.print("[yellow]No checkpoints to clean[/yellow]")
        return

    if not force:
        if not click.confirm(f"Remove all checkpoints in {pipeline_dir}?"):
            raise click.Abort()

    shutil.rmtree(pipeline_path)
    console.print(f"[green]Removed {pipeline_dir}[/green]")


@main.command()
@click.argument("schema_file", type=click.Path(exists=True))
@click.option(
    "--fix", "-f",
    is_flag=True,
    help="Interactively apply auto-fixes",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Save report to file (supports .md, .tex, .json)",
)
@click.option(
    "--sample-size",
    type=int,
    default=1000,
    help="Sample size for data-level checks",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Fail on warnings (not just errors)",
)
def audit(
    schema_file: str,
    fix: bool,
    output: Optional[str],
    sample_size: int,
    strict: bool,
) -> None:
    """
    Audit data quality for a fusion schema.

    Runs quality checks on all data sources and reports issues.

    Example:
        geopipe audit schema.yaml
        geopipe audit schema.yaml --fix
        geopipe audit schema.yaml --output quality_report.md
    """
    from geopipe.fusion.schema import FusionSchema

    console.print(f"[blue]Auditing schema: {schema_file}[/blue]")

    try:
        schema = FusionSchema.from_yaml(schema_file)
    except Exception as e:
        console.print(f"[red]Error loading schema: {e}[/red]")
        raise click.Abort()

    # Run audit
    report = schema.audit(sample_size=sample_size)

    # Display summary
    console.print(report.summary())

    # Save output if requested
    if output:
        output_path = Path(output)
        suffix = output_path.suffix.lower()

        if suffix == ".md":
            report.to_markdown(output_path)
            console.print(f"[green]Report saved to {output}[/green]")
        elif suffix == ".tex":
            output_path.write_text(report.to_latex())
            console.print(f"[green]LaTeX table saved to {output}[/green]")
        elif suffix == ".json":
            import json
            output_path.write_text(json.dumps(report.to_dict(), indent=2))
            console.print(f"[green]JSON report saved to {output}[/green]")
        else:
            report.to_markdown(output_path)
            console.print(f"[green]Report saved to {output}[/green]")

    # Apply fixes if requested
    if fix and report.fixable_issues:
        console.print(f"\n[yellow]Found {len(report.fixable_issues)} fixable issue(s)[/yellow]")
        report.apply_fixes(interactive=True)

    # Exit with error if issues found
    if report.has_errors:
        console.print("\n[red]Errors found - schema may not execute correctly[/red]")
        raise click.Abort()

    if strict and report.has_warnings:
        console.print("\n[yellow]Warnings found (--strict mode)[/yellow]")
        raise click.Abort()


@main.group()
def sources() -> None:
    """Manage data source configurations."""
    pass


@sources.command("list")
@click.argument("schema_file", type=click.Path(exists=True))
def list_sources(schema_file: str) -> None:
    """List data sources in a schema."""
    from geopipe.fusion.schema import FusionSchema

    schema = FusionSchema.from_yaml(schema_file)

    table = Table(title=f"Sources in {schema.name}")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Path", style="dim")
    table.add_column("Files", justify="right")

    for source in schema.sources:
        source_type = source.__class__.__name__.replace("Source", "").lower()
        files = source.list_files()
        table.add_row(
            source.name,
            source_type,
            source.path,
            str(len(files)),
        )

    console.print(table)


@main.group()
def specs() -> None:
    """Manage robustness specifications."""
    pass


@specs.command("list")
@click.argument("schema_file", type=click.Path(exists=True))
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show all parameters for each spec",
)
def list_specs(schema_file: str, verbose: bool) -> None:
    """
    Preview specifications that would be generated from a robustness block.

    Shows the Cartesian product of all dimensions, minus exclusions.

    Example:
        geopipe specs list schema.yaml
        geopipe specs list schema.yaml --verbose
    """
    from geopipe.fusion.schema import FusionSchema

    schema = FusionSchema.from_yaml(schema_file)

    if not schema.robustness:
        console.print("[yellow]No robustness block defined in schema[/yellow]")
        console.print("Add a 'robustness' section to generate specifications.")
        return

    spec_pairs = schema.expand_specs()

    # Summary
    console.print(f"[blue]Schema: {schema.name}[/blue]")
    console.print(f"[green]Total specifications: {len(spec_pairs)}[/green]\n")

    # Dimensions summary
    dimensions = schema.robustness.get("dimensions", {})
    if dimensions:
        dim_table = Table(title="Dimensions")
        dim_table.add_column("Dimension", style="cyan")
        dim_table.add_column("Values", style="dim")
        dim_table.add_column("Count", justify="right")

        for dim_name, values in dimensions.items():
            dim_table.add_row(dim_name, str(values), str(len(values)))

        console.print(dim_table)
        console.print()

    # Exclusions
    exclusions = schema.robustness.get("exclude", [])
    if exclusions:
        console.print(f"[yellow]Exclusions: {len(exclusions)}[/yellow]")
        for exc in exclusions:
            console.print(f"  - {exc}")
        console.print()

    # Named specs
    named = schema.robustness.get("named", {})
    if named:
        console.print(f"[green]Named specifications: {list(named.keys())}[/green]\n")

    # Spec list
    table = Table(title="Specifications")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Name", style="cyan")

    if verbose:
        # Add columns for each parameter
        sample_spec = spec_pairs[0][0] if spec_pairs else None
        if sample_spec and sample_spec.params:
            for param in sample_spec.params.keys():
                table.add_column(param, style="dim")

    for i, (spec, _) in enumerate(spec_pairs, 1):
        row = [str(i), spec.name]
        if verbose and spec.params:
            row.extend([str(v) for v in spec.params.values()])
        table.add_row(*row)

    console.print(table)


@specs.command("expand")
@click.argument("schema_file", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="schemas/",
    help="Directory for expanded schema files",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Output format for schema files",
)
def expand_specs(schema_file: str, output_dir: str, format: str) -> None:
    """
    Write individual schema files for each specification.

    Expands the robustness block into separate schema files, one per spec.

    Example:
        geopipe specs expand schema.yaml
        geopipe specs expand schema.yaml -o robustness_schemas/
    """
    import json
    import yaml

    from geopipe.fusion.schema import FusionSchema

    schema = FusionSchema.from_yaml(schema_file)

    if not schema.robustness:
        console.print("[yellow]No robustness block defined in schema[/yellow]")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    spec_pairs = schema.expand_specs()

    console.print(f"[blue]Expanding {len(spec_pairs)} specifications...[/blue]")

    for spec, configured_schema in spec_pairs:
        # Remove robustness block from output (already expanded)
        output_config = {
            "name": f"{configured_schema.name}_{spec.name}",
            "resolution": configured_schema.resolution,
            "temporal_range": list(configured_schema.temporal_range) if configured_schema.temporal_range else None,
            "sources": [s.to_dict() for s in configured_schema.sources],
            "output": configured_schema.output,
            "target_crs": configured_schema.target_crs,
        }

        if format == "yaml":
            file_path = output_path / f"{spec.name}.yaml"
            with open(file_path, "w") as f:
                yaml.dump(output_config, f, default_flow_style=False, sort_keys=False)
        else:
            file_path = output_path / f"{spec.name}.json"
            with open(file_path, "w") as f:
                json.dump(output_config, f, indent=2)

        console.print(f"  [green]✓[/green] {file_path}")

    console.print(f"\n[green]Wrote {len(spec_pairs)} schema files to {output_dir}[/green]")


@specs.command("curve")
@click.argument("results_pattern")
@click.option(
    "--estimate-col", "-e",
    default="estimate",
    help="Column name for point estimates",
)
@click.option(
    "--se-col", "-s",
    default="std_error",
    help="Column name for standard errors",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Save plot to file",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["table", "markdown", "latex"]),
    default="table",
    help="Output format for summary",
)
@click.option(
    "--schema-file",
    type=click.Path(exists=True),
    help="Schema file to get specs from (optional)",
)
def curve(
    results_pattern: str,
    estimate_col: str,
    se_col: str,
    output: Optional[str],
    format: str,
    schema_file: Optional[str],
) -> None:
    """
    Generate specification curve from results.

    Analyzes estimates across specifications and produces summary statistics
    and visualizations.

    RESULTS_PATTERN is a glob pattern for result files, e.g., "results/*/estimates.csv"

    Example:
        geopipe specs curve "results/*/estimates.csv"
        geopipe specs curve "results/*.parquet" --output curve.pdf
        geopipe specs curve "results/*.csv" --format latex
    """
    from glob import glob

    from geopipe.specs.curve import SpecificationCurve
    from geopipe.specs.variants import Spec

    # Get specs from schema or infer from files
    if schema_file:
        from geopipe.fusion.schema import FusionSchema
        schema = FusionSchema.from_yaml(schema_file)
        spec_pairs = schema.expand_specs()
        specs = [spec for spec, _ in spec_pairs]
    else:
        # Infer specs from result files
        matches = glob(results_pattern)
        if not matches:
            console.print(f"[red]No files found matching pattern: {results_pattern}[/red]")
            raise click.Abort()

        specs = []
        for match in matches:
            # Extract spec name from path
            parts = Path(match).parts
            # Try to find a meaningful name (parent directory name)
            spec_name = parts[-2] if len(parts) > 1 else Path(match).stem
            specs.append(Spec(spec_name))

    if not specs:
        console.print("[yellow]No specifications found[/yellow]")
        return

    # Create curve analyzer
    curve_analyzer = SpecificationCurve(
        specs=specs,
        results_pattern=results_pattern,
        estimate_col=estimate_col,
        se_col=se_col,
    )

    # Load and analyze
    results = curve_analyzer.load_results()

    if results.empty:
        console.print("[yellow]No results loaded - check pattern and column names[/yellow]")
        return

    console.print(f"[green]Loaded {len(results)} specification results[/green]\n")

    # Output based on format
    if format == "table":
        curve_analyzer.display()
    elif format == "markdown":
        md = curve_analyzer.to_markdown()
        console.print(md)
    elif format == "latex":
        latex = curve_analyzer.to_latex()
        console.print(latex)

    # Generate plot if requested
    if output:
        fig = curve_analyzer.plot(output)
        if fig:
            console.print(f"\n[green]Saved plot to {output}[/green]")


@specs.command("run")
@click.argument("schema_file", type=click.Path(exists=True))
@click.option(
    "--spec", "-s",
    multiple=True,
    help="Run only specific spec(s) by name",
)
@click.option(
    "--parallel", "-p",
    type=int,
    default=1,
    help="Number of parallel executions",
)
@click.option(
    "--dry-run", "-n",
    is_flag=True,
    help="Show what would be run without executing",
)
def run_specs(
    schema_file: str,
    spec: tuple[str, ...],
    parallel: int,
    dry_run: bool,
) -> None:
    """
    Run fusion for all or selected specifications.

    Executes the schema for each specification variant.

    Example:
        geopipe specs run schema.yaml
        geopipe specs run schema.yaml --spec MAIN --spec buffer_10km
        geopipe specs run schema.yaml --parallel 4
    """
    from geopipe.fusion.schema import FusionSchema

    schema = FusionSchema.from_yaml(schema_file)
    spec_pairs = schema.expand_specs()

    # Filter to specific specs if requested
    if spec:
        spec_names = set(spec)
        spec_pairs = [(s, sch) for s, sch in spec_pairs if s.name in spec_names]

        if not spec_pairs:
            console.print(f"[red]No matching specs found for: {spec}[/red]")
            raise click.Abort()

    console.print(f"[blue]Running {len(spec_pairs)} specification(s)...[/blue]\n")

    if dry_run:
        for s, sch in spec_pairs:
            console.print(f"  [dim]Would run:[/dim] {s.name} → {sch.output}")
        return

    # Sequential execution (parallel would require multiprocessing)
    results = []
    for s, configured_schema in spec_pairs:
        console.print(f"[cyan]Running spec: {s.name}[/cyan]")
        try:
            result = configured_schema.execute(show_progress=True)
            results.append((s.name, len(result), None))
            console.print(f"  [green]✓[/green] {len(result)} records → {configured_schema.output}\n")
        except Exception as e:
            results.append((s.name, 0, str(e)))
            console.print(f"  [red]✗[/red] Error: {e}\n")

    # Summary table
    console.print()
    summary_table = Table(title="Execution Summary")
    summary_table.add_column("Spec", style="cyan")
    summary_table.add_column("Records", justify="right")
    summary_table.add_column("Status", style="green")

    for name, count, error in results:
        status = f"[red]{error}[/red]" if error else "[green]✓[/green]"
        summary_table.add_row(name, str(count), status)

    console.print(summary_table)


@main.group()
def discover() -> None:
    """Discover available geospatial data."""
    pass


@discover.command("search")
@click.option(
    "--bounds", "-b",
    type=str,
    help="Bounding box: minx,miny,maxx,maxy (e.g., -122.5,37.7,-122.4,37.8)",
)
@click.option(
    "--temporal", "-t",
    type=str,
    help="Temporal range: start,end (e.g., 2020-01-01,2020-12-31)",
)
@click.option(
    "--category", "-c",
    multiple=True,
    type=click.Choice([
        "nightlights", "optical", "sar", "elevation",
        "climate", "land_cover", "vegetation", "population",
        "socioeconomic", "infrastructure",
    ]),
    help="Filter by category (can specify multiple)",
)
@click.option(
    "--provider", "-p",
    multiple=True,
    type=click.Choice(["earthengine", "planetary_computer", "stac"]),
    help="Filter by provider (can specify multiple)",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.option(
    "--max-results", "-n",
    type=int,
    default=20,
    help="Maximum number of results",
)
def discover_search(
    bounds: Optional[str],
    temporal: Optional[str],
    category: tuple[str, ...],
    provider: tuple[str, ...],
    format: str,
    max_results: int,
) -> None:
    """
    Search for available geospatial datasets.

    Queries the known datasets registry to find datasets matching
    your criteria.

    Example:
        geopipe discover search --category nightlights
        geopipe discover search -b "-122.5,37.7,-122.4,37.8" -c optical
        geopipe discover search --provider earthengine --format json
    """
    from geopipe.discovery import discover as do_discover

    # Parse bounds
    parsed_bounds = None
    if bounds:
        try:
            parts = [float(x.strip()) for x in bounds.split(",")]
            if len(parts) != 4:
                raise ValueError("Bounds must have 4 values")
            parsed_bounds = tuple(parts)  # type: ignore
        except ValueError as e:
            console.print(f"[red]Invalid bounds format: {e}[/red]")
            raise click.Abort()

    # Parse temporal range
    parsed_temporal = None
    if temporal:
        try:
            parts = [x.strip() for x in temporal.split(",")]
            if len(parts) != 2:
                raise ValueError("Temporal range must have 2 values")
            parsed_temporal = tuple(parts)  # type: ignore
        except ValueError as e:
            console.print(f"[red]Invalid temporal format: {e}[/red]")
            raise click.Abort()

    # Convert tuples to lists for filtering
    categories = list(category) if category else None
    providers = list(provider) if provider else None

    # Run discovery
    results = do_discover(
        bounds=parsed_bounds,  # type: ignore
        temporal_range=parsed_temporal,  # type: ignore
        categories=categories,  # type: ignore
        providers=providers,  # type: ignore
        max_results=max_results,
    )

    if not results:
        console.print("[yellow]No datasets found matching criteria[/yellow]")
        return

    console.print(f"[green]Found {len(results)} dataset(s)[/green]\n")

    if format == "table":
        table = Table(title="Discovered Datasets")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Provider", style="green")
        table.add_column("Resolution", justify="right")
        table.add_column("Categories", style="dim")

        for result in results:
            table.add_row(
                result.dataset_id,
                result.name,
                result.provider,
                f"{result.resolution_m}m",
                ", ".join(result.categories),
            )

        console.print(table)

    elif format == "json":
        import json
        data = [r.model_dump() for r in results]
        console.print(json.dumps(data, indent=2, default=str))

    elif format == "yaml":
        import yaml
        data = [r.model_dump() for r in results]
        console.print(yaml.dump(data, default_flow_style=False))


@discover.command("list-datasets")
@click.option(
    "--category", "-c",
    type=click.Choice([
        "nightlights", "optical", "sar", "elevation",
        "climate", "land_cover", "vegetation", "population",
        "socioeconomic", "infrastructure",
    ]),
    help="Filter by category",
)
@click.option(
    "--provider", "-p",
    type=click.Choice(["earthengine", "planetary_computer", "stac"]),
    help="Filter by provider",
)
def list_datasets(
    category: Optional[str],
    provider: Optional[str],
) -> None:
    """
    List all known datasets in the registry.

    Shows all curated datasets that can be used for discovery and fusion.

    Example:
        geopipe discover list-datasets
        geopipe discover list-datasets --category nightlights
        geopipe discover list-datasets --provider earthengine
    """
    from geopipe.discovery import CatalogRegistry

    registry = CatalogRegistry()
    registry.load()

    # Filter
    categories = [category] if category else None
    datasets = registry.filter(categories=categories, provider=provider)  # type: ignore

    if not datasets:
        console.print("[yellow]No datasets found[/yellow]")
        return

    console.print(f"[green]Found {len(datasets)} dataset(s)[/green]\n")
    registry.display(datasets)


@discover.command("info")
@click.argument("dataset_id")
def dataset_info(dataset_id: str) -> None:
    """
    Show detailed information about a specific dataset.

    DATASET_ID is the unique identifier for the dataset.

    Example:
        geopipe discover info viirs_dnb_monthly
        geopipe discover info sentinel2_l2a
    """
    from geopipe.discovery import CatalogRegistry

    registry = CatalogRegistry()
    registry.load()

    dataset = registry.get(dataset_id)

    if not dataset:
        console.print(f"[red]Dataset not found: {dataset_id}[/red]")
        console.print("\nAvailable datasets:")
        for ds in registry.list_all()[:10]:
            console.print(f"  - {ds.id}")
        raise click.Abort()

    # Display detailed info
    console.print(f"[bold cyan]{dataset.name}[/bold cyan]")
    console.print(f"[dim]ID: {dataset.id}[/dim]\n")

    info_table = Table(show_header=False, box=None)
    info_table.add_column("Field", style="cyan")
    info_table.add_column("Value")

    info_table.add_row("Provider", dataset.provider)
    info_table.add_row("Collection", dataset.collection)
    info_table.add_row("Resolution", f"{dataset.spatial_resolution_m}m")

    temporal = f"{dataset.temporal_range[0]} to "
    temporal += dataset.temporal_range[1] if dataset.temporal_range[1] else "ongoing"
    info_table.add_row("Temporal", temporal)

    info_table.add_row("Categories", ", ".join(dataset.categories))
    info_table.add_row("Bands", ", ".join(dataset.bands))

    if dataset.complementary:
        info_table.add_row("Complementary", ", ".join(dataset.complementary))

    console.print(info_table)

    if dataset.description:
        console.print(f"\n[dim]{dataset.description}[/dim]")


@discover.command("categories")
def list_categories() -> None:
    """
    List all available dataset categories.

    Shows the thematic categories used to organize datasets.
    """
    from geopipe.discovery import CatalogRegistry

    registry = CatalogRegistry()
    registry.load()

    categories = registry.list_categories()

    console.print("[bold]Available Categories[/bold]\n")

    # Count datasets per category
    for cat in categories:
        count = len(registry.filter(categories=[cat]))  # type: ignore
        console.print(f"  [cyan]{cat}[/cyan]: {count} dataset(s)")


if __name__ == "__main__":
    main()
