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

from hatchling.metadata.core import ProjectMetadata
from hatchling.plugin.manager import PluginManager
from packaging.requirements import Requirement
from packaging.version import Version


class Pyproject:
    def __init__(self, content: dict[str, Any], path: Path) -> None:
        self._content = content
        self._path = path

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as fp:
            content = tomllib.load(fp)
        return cls(content, path)

    @property
    def name(self) -> str:
        if "project" not in self._content:
            raise RuntimeError("Not a valid pyproject.toml, missing 'project' table")
        if "name" not in self._content["project"]:
            raise RuntimeError("Not a valid pyproject.toml, missing 'project.name' value")
        return self._content["project"]["name"]

    @property
    def version(self) -> Version:
        metadata = ProjectMetadata(str(self._path.parent), PluginManager())
        return Version(metadata.version)

    @property
    def requirements(self) -> list[Requirement]:
        return [Requirement(dep) for dep in self._content.get("project", {}).get("dependencies", [])]

    @property
    def members(self) -> list[str]:
        return self._content.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
