"""Robustness DSL parser and expander for geopipe."""

from __future__ import annotations

import copy
import itertools
import re
from typing import Any, Iterator

from geopipe.specs.variants import Spec


class RobustnessDSL:
    """
    Parser and expander for declarative robustness specifications.

    Parses a robustness YAML block and generates the Cartesian product
    of all specification dimensions, supporting exclusions and named specs.

    Attributes:
        dimensions: Dictionary mapping dimension names to lists of values
        exclude: List of exclusion patterns (dicts with dimension: value pairs)
        named: Dictionary of named specs with explicit values

    Example:
        >>> dsl = RobustnessDSL({
        ...     "dimensions": {
        ...         "buffer_km": [5, 10, 25],
        ...         "include_ntl": [True, False],
        ...     },
        ...     "exclude": [
        ...         {"buffer_km": 25, "include_ntl": False},
        ...     ],
        ...     "named": {
        ...         "MAIN": {"buffer_km": 10, "include_ntl": True},
        ...     },
        ... })
        >>> specs = dsl.expand()
        >>> print(len(specs))  # 5 (6 combinations minus 1 exclusion)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize RobustnessDSL from configuration.

        Args:
            config: Dictionary with 'dimensions', optionally 'exclude' and 'named'
        """
        self.dimensions: dict[str, list[Any]] = config.get("dimensions", {})
        self.exclude: list[dict[str, Any]] = config.get("exclude", [])
        self.named: dict[str, dict[str, Any]] = config.get("named", {})
        self._naming_pattern: str = config.get("naming_pattern", "")

    @classmethod
    def from_yaml_block(cls, robustness_block: dict[str, Any]) -> "RobustnessDSL":
        """
        Create RobustnessDSL from a YAML robustness block.

        Args:
            robustness_block: Dictionary from YAML parsing

        Returns:
            Configured RobustnessDSL instance
        """
        return cls(robustness_block)

    def count(self) -> int:
        """
        Count total specifications that would be generated.

        Returns:
            Number of specs (Cartesian product minus exclusions)
        """
        if not self.dimensions:
            return 0

        total = 1
        for values in self.dimensions.values():
            total *= len(values)

        # Subtract exclusions
        excluded = sum(1 for _ in self._generate_excluded())

        return max(0, total - excluded)

    def _generate_excluded(self) -> Iterator[tuple[Any, ...]]:
        """Generate tuples of excluded combinations."""
        dim_names = list(self.dimensions.keys())

        for exclude_pattern in self.exclude:
            # Generate all combinations that match this exclusion pattern
            constrained_dims = {}
            for dim_name in dim_names:
                if dim_name in exclude_pattern:
                    constrained_dims[dim_name] = [exclude_pattern[dim_name]]
                else:
                    constrained_dims[dim_name] = self.dimensions[dim_name]

            for combo in itertools.product(*[constrained_dims[d] for d in dim_names]):
                yield combo

    def _is_excluded(self, params: dict[str, Any]) -> bool:
        """Check if a parameter combination matches any exclusion pattern."""
        for exclude_pattern in self.exclude:
            matches = True
            for key, value in exclude_pattern.items():
                if key in params and params[key] != value:
                    matches = False
                    break
            if matches:
                return True
        return False

    def _generate_name(self, params: dict[str, Any]) -> str:
        """Generate a spec name from parameters."""
        # Check if this matches a named spec
        for name, named_params in self.named.items():
            if all(params.get(k) == v for k, v in named_params.items()):
                return name

        # Use naming pattern if provided
        if self._naming_pattern:
            try:
                return self._naming_pattern.format(**params)
            except KeyError:
                pass

        # Generate automatic name from parameters
        parts = []
        for key, value in sorted(params.items()):
            if isinstance(value, bool):
                if value:
                    parts.append(key)
            else:
                parts.append(f"{key}_{value}")

        return "_".join(parts) if parts else "spec"

    def expand(self) -> list[Spec]:
        """
        Generate all specification combinations.

        Returns:
            List of Spec objects for each valid parameter combination
        """
        if not self.dimensions:
            return []

        specs = []
        dim_names = list(self.dimensions.keys())
        dim_values = [self.dimensions[d] for d in dim_names]

        for combo in itertools.product(*dim_values):
            params = dict(zip(dim_names, combo))

            # Skip excluded combinations
            if self._is_excluded(params):
                continue

            name = self._generate_name(params)
            specs.append(Spec(name=name, **params))

        return specs

    def get_spec(self, name: str) -> Spec | None:
        """
        Get a named specification.

        Args:
            name: Spec name to look up

        Returns:
            Spec if found, None otherwise
        """
        if name in self.named:
            return Spec(name=name, **self.named[name])

        # Search expanded specs
        for spec in self.expand():
            if spec.name == name:
                return spec

        return None

    def apply_to_schema(
        self,
        schema_dict: dict[str, Any],
        spec: Spec,
    ) -> dict[str, Any]:
        """
        Apply spec parameters via template substitution.

        Substitutes ${param} and {param} placeholders in the schema
        with values from the spec.

        Args:
            schema_dict: Schema as dictionary (deep copied)
            spec: Spec with parameter values

        Returns:
            New schema dict with substitutions applied
        """
        result = copy.deepcopy(schema_dict)
        params = spec.params.copy()
        params["spec_name"] = spec.name

        def substitute(obj: Any) -> Any:
            """Recursively substitute templates in an object."""
            if isinstance(obj, str):
                return self._substitute_string(obj, params)
            elif isinstance(obj, dict):
                return {k: substitute(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute(item) for item in obj]
            else:
                return obj

        return substitute(result)

    def _substitute_string(self, template: str, params: dict[str, Any]) -> str:
        """
        Substitute ${param} and {param} patterns in a string.

        Args:
            template: String with potential templates
            params: Parameter values

        Returns:
            String with substitutions applied
        """
        result = template

        # Handle ${param} syntax
        for key, value in params.items():
            pattern = r"\$\{" + re.escape(key) + r"\}"
            result = re.sub(pattern, str(value), result)

        # Handle {param} syntax (but not {spec_name} which is special)
        for key, value in params.items():
            pattern = r"\{" + re.escape(key) + r"\}"
            result = re.sub(pattern, str(value), result)

        return result

    def summary(self) -> str:
        """Generate summary of robustness configuration."""
        lines = [
            f"RobustnessDSL Configuration",
            f"  Dimensions: {len(self.dimensions)}",
        ]

        for dim_name, values in self.dimensions.items():
            lines.append(f"    {dim_name}: {values}")

        lines.append(f"  Exclusions: {len(self.exclude)}")
        lines.append(f"  Named specs: {list(self.named.keys())}")
        lines.append(f"  Total specs: {self.count()}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"RobustnessDSL(dimensions={list(self.dimensions.keys())}, count={self.count()})"


def expand_robustness_specs(
    schema_dict: dict[str, Any],
) -> tuple[list[Spec], list[dict[str, Any]]]:
    """
    Expand a schema with robustness block into specs and configured schemas.

    Args:
        schema_dict: Schema dictionary with optional 'robustness' block

    Returns:
        Tuple of (list of Specs, list of configured schema dicts)
    """
    robustness_block = schema_dict.get("robustness")

    if not robustness_block:
        # No robustness block - return single "MAIN" spec
        return [Spec("MAIN")], [schema_dict]

    dsl = RobustnessDSL.from_yaml_block(robustness_block)
    specs = dsl.expand()

    # Remove robustness block from schema template
    base_schema = {k: v for k, v in schema_dict.items() if k != "robustness"}

    # Generate configured schemas
    schemas = [dsl.apply_to_schema(base_schema, spec) for spec in specs]

    return specs, schemas
