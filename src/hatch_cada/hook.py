import warnings
from fnmatch import fnmatch
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface
from hatchling.plugin import hookimpl

from hatch_cada.constants import PYPROJECT_NAME, UV_LOCKFILE_NAME
from hatch_cada.lockfile import Lockfile
from hatch_cada.pyproject import Pyproject
from hatch_cada.strategy import Strategy
from hatch_cada.utils import find_workspace_root


class CadaMetaHook(MetadataHookInterface):
    """Metadata hook that rewrites workspace dependencies to versioned dependencies."""

    PLUGIN_NAME = "cada"

    # Packages currently being resolved. When resolving an editable dependency's
    # version, hatchling may trigger this hook again for that dependency. If it
    # in turn depends back on a package already being resolved, we'd recurse
    # infinitely. Skipping re-entrant calls is safe because the caller only
    # needs the version, which is resolved by the version source plugin
    # independently of metadata hooks.
    _resolving: set[str] = set()

    def update(self, metadata: dict) -> None:
        """Rewrite workspace dependencies with version constraints.

        Args:
            metadata: The metadata dict to update with rewritten dependencies.

        Raises:
            ValueError: If strategy is missing or invalid.
        """
        pkg_name = metadata.get("name", "")
        if pkg_name in self._resolving:
            return

        self._resolving.add(pkg_name)
        try:
            if "strategy" not in self.config:
                valid = ", ".join(s.value for s in Strategy)
                raise ValueError(
                    f"Missing required 'strategy' option. "
                    f"Add to pyproject.toml:\n\n"
                    f"[tool.hatch.metadata.hooks.cada]\n"
                    f'strategy = "allow-all-updates"\n\n'
                    f"Valid strategies: {valid}"
                )

            default_strategy = Strategy.from_string(self.config["strategy"])
            overrides = {name: Strategy.from_string(value) for name, value in self.config.get("overrides", {}).items()}

            pkg_pyproject = Pyproject.load(Path(self.root) / PYPROJECT_NAME)
            dependencies = pkg_pyproject.dependencies
            optional_dependencies = pkg_pyproject.optional_dependencies

            workspace_root = find_workspace_root(Path(self.root))
            if workspace_root is None:
                warnings.warn(
                    f"No workspace found for {self.root}. "
                    f"Workspace dependencies will not be rewritten. "
                    f"Set WORKSPACE_ROOT or ensure uv.lock exists in a parent directory.",
                    stacklevel=2,
                )
                return

            workspace_pyproject = Pyproject.load(workspace_root / PYPROJECT_NAME)
            lockfile = Lockfile.load(workspace_root / UV_LOCKFILE_NAME)
            member_patterns = workspace_pyproject.members

            if dependencies:
                new_deps: list[str] = []
                for req in dependencies:
                    pkg = lockfile.get_package(req.name)
                    if pkg and pkg.editable_path and any(fnmatch(pkg.editable_path, p) for p in member_patterns):
                        strategy = overrides.get(req.name, default_strategy)
                        req.specifier = strategy.make_specifier(pkg.version)
                    new_deps.append(str(req))
                metadata["dependencies"] = new_deps

            if optional_dependencies:
                new_opt_deps: dict[str, list[str]] = {}
                for group, reqs in optional_dependencies.items():
                    new_group_deps: list[str] = []
                    for req in reqs:
                        pkg = lockfile.get_package(req.name)
                        if pkg and pkg.editable_path and any(fnmatch(pkg.editable_path, p) for p in member_patterns):
                            strategy = overrides.get(req.name, default_strategy)
                            req.specifier = strategy.make_specifier(pkg.version)
                        new_group_deps.append(str(req))
                    new_opt_deps[group] = new_group_deps
                metadata["optional-dependencies"] = new_opt_deps
        finally:
            self._resolving.discard(pkg_name)


@hookimpl
def hatch_register_metadata_hook() -> type[CadaMetaHook]:  # pragma: no cover
    return CadaMetaHook
