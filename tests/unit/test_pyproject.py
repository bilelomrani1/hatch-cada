import textwrap
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from hatch_cada.pyproject import Pyproject


class TestPyproject:
    def test_load_from_path(self, tmp_path: Path) -> None:
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text('[project]\nname = "foo"\nversion = "1.0.0"\n')

        pyproject = Pyproject.load(pyproject_path)

        assert pyproject.name == "foo"


class TestName:
    def test_returns_project_name(self, tmp_path: Path) -> None:
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text('[project]\nname = "mypackage"\nversion = "1.0.0"\n')

        pyproject = Pyproject.load(pyproject_path)

        assert pyproject.name == "mypackage"

    def test_raises_for_missing_project_table(self, tmp_path: Path) -> None:
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text("[tool.something]\nkey = 1\n")

        pyproject = Pyproject.load(pyproject_path)

        with pytest.raises(RuntimeError, match="missing 'project' table"):
            _ = pyproject.name

    def test_raises_for_missing_name(self, tmp_path: Path) -> None:
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text('[project]\nversion = "1.0.0"\n')

        pyproject = Pyproject.load(pyproject_path)

        with pytest.raises(RuntimeError, match="missing 'project.name' value"):
            _ = pyproject.name


class TestVersion:
    def test_returns_version(self, tmp_path: Path) -> None:
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text('[project]\nname = "foo"\nversion = "2.5.0"\n')

        pyproject = Pyproject.load(pyproject_path)

        assert pyproject.version == Version("2.5.0")

    def test_raises_for_missing_version(self, tmp_path: Path) -> None:
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text('[project]\nname = "foo"\n')

        pyproject = Pyproject.load(pyproject_path)

        with pytest.raises(ValueError, match="can only be resolved dynamically"):
            _ = pyproject.version


class TestRequirements:
    def test_returns_requirements(self, tmp_path: Path) -> None:
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(
            textwrap.dedent("""
                [project]
                name = "foo"
                version = "1.0.0"
                dependencies = ["requests>=2.0", "click"]
            """)
        )

        pyproject = Pyproject.load(pyproject_path)

        assert pyproject.requirements == [Requirement("requests>=2.0"), Requirement("click")]

    def test_returns_empty_list_when_no_dependencies(self, tmp_path: Path) -> None:
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text('[project]\nname = "foo"\nversion = "1.0.0"\n')

        pyproject = Pyproject.load(pyproject_path)

        assert pyproject.requirements == []


class TestMembers:
    def test_returns_workspace_members(self, tmp_path: Path) -> None:
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(
            textwrap.dedent("""
                [project]
                name = "workspace"
                version = "1.0.0"

                [tool.uv.workspace]
                members = ["packages/*"]
            """)
        )

        pyproject = Pyproject.load(pyproject_path)

        assert pyproject.members == ["packages/*"]

    def test_returns_empty_list_when_no_workspace(self, tmp_path: Path) -> None:
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text('[project]\nname = "foo"\nversion = "1.0.0"\n')

        pyproject = Pyproject.load(pyproject_path)

        assert pyproject.members == []
