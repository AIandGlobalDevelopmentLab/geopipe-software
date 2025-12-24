"""Shared utilities for remote data sources."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


class RemoteSourceMixin:
    """
    Mixin class providing common functionality for remote data sources.

    Provides utilities for:
    - Converting center + patch_size to bounds
    - Computing deterministic cache keys
    - Checking and managing download cache
    - Downloading files with progress indication
    """

    def _compute_bounds(
        self,
        center: tuple[float, float] | None = None,
        patch_size_km: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Compute bounding box from center + patch_size or return explicit bounds.

        Args:
            center: (longitude, latitude) center point
            patch_size_km: Size of patch in kilometers (creates square patch)
            bounds: Explicit bounds (minx, miny, maxx, maxy)

        Returns:
            Bounding box as (minx, miny, maxx, maxy) in WGS84

        Raises:
            ValueError: If neither bounds nor center+patch_size provided
        """
        if bounds is not None:
            return bounds

        if center is None or patch_size_km is None:
            raise ValueError(
                "Must provide either 'bounds' or both 'center' and 'patch_size_km'"
            )

        lon, lat = center
        # Approximate degrees per km at given latitude
        km_per_deg_lat = 111.32  # roughly constant
        km_per_deg_lon = 111.32 * math.cos(math.radians(lat))

        half_size_lat = (patch_size_km / 2) / km_per_deg_lat
        half_size_lon = (patch_size_km / 2) / km_per_deg_lon

        return (
            lon - half_size_lon,
            lat - half_size_lat,
            lon + half_size_lon,
            lat + half_size_lat,
        )

    def _compute_cache_key(
        self,
        source_type: str,
        collection: str,
        bands: list[str],
        bounds: tuple[float, float, float, float],
        temporal_range: tuple[str, str] | None,
        resolution: float,
        **extra: Any,
    ) -> str:
        """
        Compute deterministic cache key from query parameters.

        Args:
            source_type: Type of remote source (e.g., "stac", "earthengine")
            collection: Collection/dataset identifier
            bands: List of band/asset names
            bounds: Spatial bounds
            temporal_range: Time range tuple
            resolution: Target resolution
            **extra: Additional parameters to include in key

        Returns:
            Hash string suitable for use as filename
        """
        key_data = {
            "source_type": source_type,
            "collection": collection,
            "bands": sorted(bands),
            "bounds": [round(b, 6) for b in bounds],
            "temporal_range": temporal_range,
            "resolution": resolution,
            **extra,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _get_cache_dir(self, output_dir: str, source_type: str) -> Path:
        """
        Get cache directory path, creating if necessary.

        Args:
            output_dir: Base output directory
            source_type: Type of remote source

        Returns:
            Path to cache directory
        """
        cache_dir = Path(output_dir).expanduser() / source_type
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_cache_path(
        self,
        output_dir: str,
        source_type: str,
        cache_key: str,
        extension: str = ".tif",
    ) -> Path:
        """
        Get path for cached file.

        Args:
            output_dir: Base output directory
            source_type: Type of remote source
            cache_key: Hash key for this query
            extension: File extension

        Returns:
            Path where cached file should be stored
        """
        cache_dir = self._get_cache_dir(output_dir, source_type)
        return cache_dir / f"{cache_key}{extension}"

    def _check_cache(
        self,
        output_dir: str,
        source_type: str,
        cache_key: str,
        extension: str = ".tif",
    ) -> Path | None:
        """
        Check if cached file exists.

        Args:
            output_dir: Base output directory
            source_type: Type of remote source
            cache_key: Hash key for this query
            extension: File extension

        Returns:
            Path to cached file if exists, None otherwise
        """
        cache_path = self._get_cache_path(output_dir, source_type, cache_key, extension)
        if cache_path.exists():
            return cache_path
        return None

    def _save_cache_metadata(
        self,
        cache_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        """
        Save metadata alongside cached file.

        Args:
            cache_path: Path to cached data file
            metadata: Metadata dictionary to save
        """
        meta_path = cache_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _download_with_progress(
        self,
        url: str,
        output_path: Path,
        description: str = "Downloading",
        chunk_size: int = 8192,
    ) -> None:
        """
        Download file from URL with progress indication.

        Args:
            url: URL to download from
            output_path: Local path to save file
            description: Description for progress display
            chunk_size: Size of download chunks

        Raises:
            ImportError: If requests not installed
            requests.RequestException: If download fails
        """
        try:
            import requests
        except ImportError as e:
            raise ImportError(
                "requests is required for downloading. Install with: pip install requests"
            ) from e

        try:
            from rich.progress import (
                BarColumn,
                DownloadColumn,
                Progress,
                TextColumn,
                TransferSpeedColumn,
            )

            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
            ) as progress:
                task = progress.add_task(description, total=total_size)
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
        except ImportError:
            # Fallback without rich progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

    def _ensure_dependencies(self, packages: list[str], pip_install: str) -> None:
        """
        Check that required packages are installed.

        Args:
            packages: List of package names to check
            pip_install: pip install command to show in error

        Raises:
            ImportError: If any package is missing
        """
        missing = []
        for package in packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)

        if missing:
            raise ImportError(
                f"Missing required packages: {', '.join(missing)}. "
                f"Install with: {pip_install}"
            )

    def _get_env_or_config(
        self,
        config_value: str | None,
        env_var: str,
        default: str | None = None,
    ) -> str | None:
        """
        Get value from config, falling back to environment variable.

        Args:
            config_value: Value from configuration
            env_var: Environment variable name
            default: Default value if neither found

        Returns:
            Value from config, env var, or default
        """
        if config_value is not None:
            return config_value
        return os.environ.get(env_var, default)
