import textwrap
from pathlib import Path

import pytest

from hatch_cada.hook import CadaMetaHook


def create_hook(root: Path, config: dict | None = None) -> CadaMetaHook:
    hook = CadaMetaHook.__new__(CadaMetaHook)
    hook._MetadataHookInterface__root = str(root)
    hook._MetadataHookInterface__config = config or {}
    return hook


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    # Create workspace pyproject.toml with both hardcoded and glob members
    (workspace_root / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "workspace"
            version = "0.0.0"

            [tool.uv.workspace]
            members = ["main", "packages/*"]
        """)
    )

    # Create main package (hardcoded member)
    main_pkg = workspace_root / "main"
    main_pkg.mkdir()
    (main_pkg / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "main"
            version = "1.0.0"
            dependencies = ["dep"]
        """)
    )

    # Create dep package (glob member under packages/)
    packages_dir = workspace_root / "packages"
    packages_dir.mkdir()
    dep_pkg = packages_dir / "dep"
    dep_pkg.mkdir()
    (dep_pkg / "pyproject.toml").write_text('[project]\nname = "dep"\nversion = "2.0.0"\n')

    # Create lockfile
    lock_content = textwrap.dedent("""
        version = 1
        requires-python = ">=3.12"

        [[package]]
        name = "dep"
        version = "2.0.0"
        source = { editable = "packages/dep" }

        [[package]]
        name = "main"
        version = "1.0.0"
        source = { editable = "main" }
    """)
    (workspace_root / "uv.lock").write_text(lock_content)

    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace_root))

    return workspace_root


class TestCadaMetaHook:
    def test_raises_when_strategy_missing(self, workspace: Path) -> None:
        hook = create_hook(workspace / "main", {})
        metadata = {"name": "main"}

        with pytest.raises(ValueError, match="Missing required 'strategy' option"):
            hook.update(metadata)

    def test_error_shows_example_config(self, workspace: Path) -> None:
        hook = create_hook(workspace / "main", {})
        metadata = {"name": "main"}

        with pytest.raises(ValueError, match=r"\[tool\.hatch\.metadata\.hooks\.cada\]"):
            hook.update(metadata)

    def test_raises_for_invalid_strategy(self, workspace: Path) -> None:
        hook = create_hook(workspace / "main", {"strategy": "invalid"})
        metadata = {"name": "main"}

        with pytest.raises(ValueError, match="Invalid strategy 'invalid'"):
            hook.update(metadata)

    @pytest.mark.parametrize(
        ("strategy", "expected_specifier"),
        [
            ("pin", "dep==2.0.0"),
            ("allow-patch-updates", "dep<2.1.0,>=2.0.0"),
            ("allow-minor-updates", "dep<3.0.0,>=2.0.0"),
            ("allow-all-updates", "dep>=2.0.0"),
            ("semver", "dep<3.0.0,>=2.0.0"),
        ],
    )
    def test_rewrites_workspace_dependency(self, workspace: Path, strategy: str, expected_specifier: str) -> None:
        hook = create_hook(workspace / "main", {"strategy": strategy})
        metadata = {"name": "main"}

        hook.update(metadata)

        assert expected_specifier in metadata["dependencies"]

    def test_preserves_non_workspace_dependencies(self, workspace: Path) -> None:
        (workspace / "main" / "pyproject.toml").write_text(
            textwrap.dedent("""
                [project]
                name = "main"
                version = "1.0.0"
                dependencies = ["dep", "requests>=2.0"]
            """)
        )
        # Add requests to lockfile
        (workspace / "uv.lock").write_text(
            textwrap.dedent("""
                version = 1
                requires-python = ">=3.12"

                [[package]]
                name = "dep"
                version = "2.0.0"
                source = { editable = "packages/dep" }

                [[package]]
                name = "main"
                version = "1.0.0"
                source = { editable = "main" }

                [[package]]
                name = "requests"
                version = "2.31.0"
            """)
        )

        hook = create_hook(workspace / "main", {"strategy": "allow-all-updates"})
        metadata = {"name": "main"}

        hook.update(metadata)

        assert "dep>=2.0.0" in metadata["dependencies"]
        assert "requests>=2.0" in metadata["dependencies"]

    def test_preserves_extras(self, workspace: Path) -> None:
        (workspace / "main" / "pyproject.toml").write_text(
            textwrap.dedent("""
                [project]
                name = "main"
                version = "1.0.0"
                dependencies = ["dep[extra1,extra2]"]
            """)
        )

        hook = create_hook(workspace / "main", {"strategy": "allow-all-updates"})
        metadata = {"name": "main"}

        hook.update(metadata)

        assert "dep[extra1,extra2]>=2.0.0" in metadata["dependencies"]

    def test_returns_early_when_no_dependencies(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WORKSPACE_ROOT", raising=False)

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "pyproject.toml").write_text('[project]\nname = "pkg"\nversion = "1.0.0"\n')

        hook = create_hook(pkg, {"strategy": "allow-all-updates"})
        metadata = {"name": "pkg"}

        hook.update(metadata)

        assert "dependencies" not in metadata

    def test_warns_when_no_workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WORKSPACE_ROOT", raising=False)

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "pyproject.toml").write_text(
            textwrap.dedent("""
                [project]
                name = "pkg"
                version = "1.0.0"
                dependencies = ["requests"]
            """)
        )

        hook = create_hook(pkg, {"strategy": "allow-all-updates"})
        metadata = {"name": "pkg"}

        with pytest.warns(UserWarning, match="No workspace found"):
            hook.update(metadata)

        assert "dependencies" not in metadata

    def test_override_single_dependency(self, workspace: Path) -> None:
        hook = create_hook(workspace / "main", {"strategy": "allow-all-updates", "overrides": {"dep": "pin"}})
        metadata = {"name": "main"}

        hook.update(metadata)

        assert "dep==2.0.0" in metadata["dependencies"]

    def test_override_with_multiple_dependencies(self, workspace: Path) -> None:
        # Add a second workspace dependency
        other_pkg = workspace / "packages" / "other"
        other_pkg.mkdir()
        (other_pkg / "pyproject.toml").write_text('[project]\nname = "other"\nversion = "3.0.0"\n')

        (workspace / "main" / "pyproject.toml").write_text(
            textwrap.dedent("""
                [project]
                name = "main"
                version = "1.0.0"
                dependencies = ["dep", "other"]
            """)
        )
        (workspace / "uv.lock").write_text(
            textwrap.dedent("""
                version = 1
                requires-python = ">=3.12"

                [[package]]
                name = "dep"
                version = "2.0.0"
                source = { editable = "packages/dep" }

                [[package]]
                name = "other"
                version = "3.0.0"
                source = { editable = "packages/other" }

                [[package]]
                name = "main"
                version = "1.0.0"
                source = { editable = "main" }
            """)
        )

        hook = create_hook(workspace / "main", {"strategy": "allow-all-updates", "overrides": {"dep": "pin"}})
        metadata = {"name": "main"}

        hook.update(metadata)

        assert "dep==2.0.0" in metadata["dependencies"]
        assert "other>=3.0.0" in metadata["dependencies"]

    def test_override_invalid_strategy_raises(self, workspace: Path) -> None:
        hook = create_hook(workspace / "main", {"strategy": "allow-all-updates", "overrides": {"dep": "invalid"}})
        metadata = {"name": "main"}

        with pytest.raises(ValueError, match="Invalid strategy 'invalid'"):
            hook.update(metadata)
