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


if __name__ == "__main__":
    main()
