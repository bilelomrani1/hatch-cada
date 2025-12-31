from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from hatch_cada.hook import CadaMetaHook


@dataclass
class ExternalPackage:
    name: str
    version: str


@dataclass
class WorkspacePackage:
    name: str
    version: str
    dependencies: list[str] = field(default_factory=list)
    optional_dependencies: dict[str, list[str]] = field(default_factory=dict)

    def pyproject(self) -> str:
        data: dict[str, Any] = {"project": {"name": self.name, "version": self.version}}
        if self.dependencies:
            data["project"]["dependencies"] = self.dependencies
        if self.optional_dependencies:
            data["project"]["optional-dependencies"] = self.optional_dependencies
        return tomli_w.dumps(data)


@dataclass
class WorkspaceConfig:
    packages: list[WorkspacePackage]
    locked_packages: list[ExternalPackage] = field(default_factory=list)
    members: list[str] | None = None

    def pyproject(self, members: list[str]) -> str:
        data = {
            "project": {"name": "workspace", "version": "0.0.0"},
            "tool": {"uv": {"workspace": {"members": members}}},
        }
        return tomli_w.dumps(data)

    def lockfile(self) -> str:
        lines = ["version = 1", 'requires-python = ">=3.12"', ""]
        for pkg in self.packages:
            lines.extend(
                [
                    "[[package]]",
                    f'name = "{pkg.name}"',
                    f'version = "{pkg.version}"',
                    f'source = {{ editable = "{pkg.name}" }}',
                    "",
                ]
            )
        for pkg in self.locked_packages:
            lines.extend(
                [
                    "[[package]]",
                    f'name = "{pkg.name}"',
                    f'version = "{pkg.version}"',
                    "",
                ]
            )
        return "\n".join(lines)


WorkspaceFactory = Callable[[WorkspaceConfig], Path]


@pytest.fixture
def workspace_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkspaceFactory:
    def _create(config: WorkspaceConfig) -> Path:
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        members = config.members if config.members is not None else [pkg.name for pkg in config.packages]
        (workspace_root / "pyproject.toml").write_text(config.pyproject(members))
        for pkg in config.packages:
            pkg_path = workspace_root / pkg.name
            pkg_path.mkdir(parents=True, exist_ok=True)
            (pkg_path / "pyproject.toml").write_text(pkg.pyproject())
        (workspace_root / "uv.lock").write_text(config.lockfile())
        monkeypatch.setenv("WORKSPACE_ROOT", str(workspace_root))
        return workspace_root

    return _create


def create_hook(root: Path, config: dict | None = None) -> CadaMetaHook:
    hook = CadaMetaHook.__new__(CadaMetaHook)
    hook._MetadataHookInterface__root = str(root)  # pyright: ignore[reportAttributeAccessIssue]
    hook._MetadataHookInterface__config = config or {}  # pyright: ignore[reportAttributeAccessIssue]
    return hook


class TestCadaMetaHook:
    def test_warns_when_no_workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WORKSPACE_ROOT", raising=False)
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "pyproject.toml").write_text('[project]\nname = "pkg"\nversion = "1.0.0"\ndependencies = ["requests"]\n')

        hook = create_hook(pkg, {"strategy": "allow-all-updates"})
        metadata = {"name": "pkg"}

        with pytest.warns(UserWarning, match="No workspace found"):
            hook.update(metadata)

        assert metadata == {"name": "pkg"}

    @pytest.mark.parametrize(
        ("hook_config", "error_match"),
        [
            pytest.param({}, "Missing required 'strategy' option", id="missing_strategy"),
            pytest.param({"strategy": "invalid"}, "Invalid strategy 'invalid'", id="invalid_strategy"),
            pytest.param(
                {"strategy": "allow-all-updates", "overrides": {"dep": "invalid"}},
                "Invalid strategy 'invalid'",
                id="invalid_override_strategy",
            ),
        ],
    )
    def test_raises_for_invalid_config(
        self, workspace_factory: WorkspaceFactory, hook_config: dict, error_match: str
    ) -> None:
        workspace = workspace_factory(
            WorkspaceConfig(
                packages=[
                    WorkspacePackage("main", "1.0.0", dependencies=["dep"]),
                    WorkspacePackage("dep", "2.0.0"),
                ]
            )
        )
        hook = create_hook(workspace / "main", hook_config)

        with pytest.raises(ValueError, match=error_match):
            hook.update({"name": "main"})

    @pytest.mark.parametrize(
        ("strategy", "dep_version", "expected_metadata"),
        [
            pytest.param("pin", "2.0.0", {"name": "main", "dependencies": ["dep==2.0.0"]}, id="pin"),
            pytest.param(
                "allow-patch-updates", "2.0.0", {"name": "main", "dependencies": ["dep<2.1.0,>=2.0.0"]}, id="patch"
            ),
            pytest.param(
                "allow-minor-updates", "2.0.0", {"name": "main", "dependencies": ["dep<3.0.0,>=2.0.0"]}, id="minor"
            ),
            pytest.param("allow-all-updates", "2.0.0", {"name": "main", "dependencies": ["dep>=2.0.0"]}, id="all"),
            pytest.param("semver", "2.0.0", {"name": "main", "dependencies": ["dep<3.0.0,>=2.0.0"]}, id="semver"),
            pytest.param("semver", "0.2.0", {"name": "main", "dependencies": ["dep<0.3.0,>=0.2.0"]}, id="semver_0x"),
        ],
    )
    def test_applies_strategy(
        self, workspace_factory: WorkspaceFactory, strategy: str, dep_version: str, expected_metadata: dict[str, Any]
    ) -> None:
        workspace = workspace_factory(
            WorkspaceConfig(
                packages=[
                    WorkspacePackage("main", "1.0.0", dependencies=["dep"]),
                    WorkspacePackage("dep", dep_version),
                ]
            )
        )
        hook = create_hook(workspace / "main", {"strategy": strategy})
        metadata: dict[str, Any] = {"name": "main"}

        hook.update(metadata)

        assert metadata == expected_metadata

    @pytest.mark.parametrize(
        ("workspace_config", "hook_config", "expected_metadata"),
        [
            pytest.param(
                WorkspaceConfig(
                    packages=[WorkspacePackage("main", "1.0.0", dependencies=["requests>=2.0"])],
                    locked_packages=[ExternalPackage("requests", "2.31.0")],
                ),
                {"strategy": "allow-all-updates"},
                {"name": "main", "dependencies": ["requests>=2.0"]},
                id="preserves_external_dep",
            ),
            pytest.param(
                WorkspaceConfig(
                    packages=[
                        WorkspacePackage("main", "1.0.0", dependencies=["dep-with-extras[extra1,extra2]"]),
                        WorkspacePackage(
                            "dep-with-extras", "4.0.0", optional_dependencies={"extra1": [], "extra2": []}
                        ),
                    ]
                ),
                {"strategy": "pin"},
                {"name": "main", "dependencies": ["dep-with-extras[extra1,extra2]==4.0.0"]},
                id="preserves_extras",
            ),
            pytest.param(
                WorkspaceConfig(
                    packages=[
                        WorkspacePackage("main", "1.0.0", dependencies=['dep ; python_version >= "3.10"']),
                        WorkspacePackage("dep", "2.0.0"),
                    ]
                ),
                {"strategy": "pin"},
                {"name": "main", "dependencies": ['dep==2.0.0; python_version >= "3.10"']},
                id="preserves_python_marker",
            ),
            pytest.param(
                WorkspaceConfig(
                    packages=[
                        WorkspacePackage("main", "1.0.0", dependencies=['dep ; sys_platform == "linux"']),
                        WorkspacePackage("dep", "2.0.0"),
                    ]
                ),
                {"strategy": "pin"},
                {"name": "main", "dependencies": ['dep==2.0.0; sys_platform == "linux"']},
                id="preserves_platform_marker",
            ),
            pytest.param(
                WorkspaceConfig(
                    packages=[
                        WorkspacePackage("main", "1.0.0", dependencies=['dep[extra1] ; python_version >= "3.11"']),
                        WorkspacePackage("dep", "2.0.0", optional_dependencies={"extra1": []}),
                    ]
                ),
                {"strategy": "pin"},
                {"name": "main", "dependencies": ['dep[extra1]==2.0.0; python_version >= "3.11"']},
                id="preserves_extras_and_marker",
            ),
            pytest.param(
                WorkspaceConfig(
                    packages=[
                        WorkspacePackage("main", "1.0.0", dependencies=["dep", "dep-with-extras[extra1,extra2]"]),
                        WorkspacePackage("dep", "2.0.0"),
                        WorkspacePackage(
                            "dep-with-extras", "4.0.0", optional_dependencies={"extra1": [], "extra2": []}
                        ),
                    ]
                ),
                {"strategy": "allow-all-updates", "overrides": {"dep": "pin"}},
                {"name": "main", "dependencies": ["dep==2.0.0", "dep-with-extras[extra1,extra2]>=4.0.0"]},
                id="override_strategy",
            ),
            pytest.param(
                WorkspaceConfig(
                    packages=[
                        WorkspacePackage("main", "1.0.0", optional_dependencies={"dev": ["opt-dep"]}),
                        WorkspacePackage("opt-dep", "3.0.0"),
                    ]
                ),
                {"strategy": "pin"},
                {"name": "main", "optional-dependencies": {"dev": ["opt-dep==3.0.0"]}},
                id="optional_deps",
            ),
            pytest.param(
                WorkspaceConfig(
                    packages=[
                        WorkspacePackage("main", "1.0.0", optional_dependencies={"dev": ["pytest>=7.0"]}),
                    ],
                    locked_packages=[ExternalPackage("pytest", "8.0.0")],
                ),
                {"strategy": "allow-all-updates"},
                {"name": "main", "optional-dependencies": {"dev": ["pytest>=7.0"]}},
                id="preserves_external_optional_dep",
            ),
            pytest.param(
                WorkspaceConfig(packages=[WorkspacePackage("main", "1.0.0")]),
                {"strategy": "allow-all-updates"},
                {"name": "main"},
                id="no_deps",
            ),
        ],
    )
    def test_rewrites_dependencies(
        self,
        workspace_factory: WorkspaceFactory,
        workspace_config: WorkspaceConfig,
        hook_config: dict,
        expected_metadata: dict[str, Any],
    ) -> None:
        workspace = workspace_factory(workspace_config)
        hook = create_hook(workspace / "main", hook_config)
        metadata: dict[str, Any] = {"name": "main"}

        hook.update(metadata)

        assert metadata == expected_metadata
