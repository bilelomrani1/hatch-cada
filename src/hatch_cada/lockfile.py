from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from typing import Self
except ImportError:  # pragma: <3.11 cover
    from typing_extensions import Self

try:
    import tomllib
except ModuleNotFoundError:  # pragma: <3.11 cover
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

from packaging.version import Version

from hatch_cada.constants import PYPROJECT_NAME
from hatch_cada.pyproject import Pyproject


@dataclass(kw_only=True, frozen=True)
class Package:
    name: str
    version: Version
    editable_path: str | None = None

    @classmethod
    def from_lock_entry(cls, entry: dict[str, Any], root: Path) -> Self:
        """Create a Package from a lockfile entry.

        Args:
            entry: The raw package entry from the lockfile.
            root: The workspace root path for resolving editable packages.

        Returns:
            The package with resolved version.

        Raises:
            KeyError: If the package has no version and is not editable.
        """
        name = entry["name"]
        source = entry.get("source", {})
        editable_path = source.get("editable")
        directory_path = source.get("directory")

        if "version" in entry:
            version = Version(entry["version"])
        elif editable_path:
            version = Pyproject.load(root / editable_path / PYPROJECT_NAME).version
        elif directory_path:
            version = Pyproject.load(root / directory_path / PYPROJECT_NAME).version
        else:
            raise KeyError(f"Package '{name}' has no version and is not editable")

        return cls(name=name, version=version, editable_path=editable_path)


class Lockfile:
    def __init__(self, content: dict[str, Any], root: Path) -> None:
        self._content = content
        self._root = root

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as fp:
            content = tomllib.load(fp)
        return cls(content, path.parent)

    @property
    def _packages(self) -> dict:
        return {pkg["name"]: pkg for pkg in self._content.get("package", [])}

    def get_package(self, name: str) -> Package | None:
        """Get a package from the lockfile.

        Args:
            name: The package name to look up.

        Returns:
            The package with resolved version, or None if not found.
            Returns None during `uv add` when the lockfile may not yet
            contain the package being added.
        """
        entry = self._packages.get(name)
        if entry is None:
            return None

        return Package.from_lock_entry(entry, self._root)
