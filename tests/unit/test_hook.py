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

    # Create main package
    main_pkg = workspace_root / "main"
    main_pkg.mkdir()
    (main_pkg / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "main"
            version = "1.0.0"
            dependencies = ["dep", "dep-with-extras[extra1,extra2]", "requests>=2.0"]

            [project.optional-dependencies]
            dev = ["opt-dep", "pytest>=7.0"]
        """)
    )

    # Create dep package
    packages_dir = workspace_root / "packages"
    packages_dir.mkdir()
    dep_pkg = packages_dir / "dep"
    dep_pkg.mkdir()
    (dep_pkg / "pyproject.toml").write_text('[project]\nname = "dep"\nversion = "2.0.0"\n')

    # Create opt-dep package
    opt_dep_pkg = packages_dir / "opt-dep"
    opt_dep_pkg.mkdir()
    (opt_dep_pkg / "pyproject.toml").write_text('[project]\nname = "opt-dep"\nversion = "3.0.0"\n')

    # Create dep-with-extras package
    dep_with_extras_pkg = packages_dir / "dep-with-extras"
    dep_with_extras_pkg.mkdir()
    (dep_with_extras_pkg / "pyproject.toml").write_text(
        textwrap.dedent("""
            [project]
            name = "dep-with-extras"
            version = "4.0.0"

            [project.optional-dependencies]
            extra1 = []
            extra2 = []
        """)
    )

    # Create lockfile
    lock_content = textwrap.dedent("""
        version = 1
        requires-python = ">=3.12"

        [[package]]
        name = "dep"
        version = "2.0.0"
        source = { editable = "packages/dep" }

        [[package]]
        name = "opt-dep"
        version = "3.0.0"
        source = { editable = "packages/opt-dep" }

        [[package]]
        name = "dep-with-extras"
        version = "4.0.0"
        source = { editable = "packages/dep-with-extras" }

        [[package]]
        name = "main"
        version = "1.0.0"
        source = { editable = "main" }

        [[package]]
        name = "requests"
        version = "2.31.0"

        [[package]]
        name = "pytest"
        version = "8.0.0"
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

    def test_preserves_non_workspace_dependency(self, workspace: Path) -> None:
        hook = create_hook(workspace / "main", {"strategy": "allow-all-updates"})
        metadata = {"name": "main"}

        hook.update(metadata)

        assert "requests>=2.0" in metadata["dependencies"]

    @pytest.mark.parametrize(
        ("strategy", "expected_specifier"),
        [
            ("pin", "opt-dep==3.0.0"),
            ("allow-patch-updates", "opt-dep<3.1.0,>=3.0.0"),
            ("allow-minor-updates", "opt-dep<4.0.0,>=3.0.0"),
            ("allow-all-updates", "opt-dep>=3.0.0"),
            ("semver", "opt-dep<4.0.0,>=3.0.0"),
        ],
    )
    def test_rewrites_workspace_optional_dependency(
        self, workspace: Path, strategy: str, expected_specifier: str
    ) -> None:
        hook = create_hook(workspace / "main", {"strategy": strategy})
        metadata = {"name": "main"}

        hook.update(metadata)

        assert "optional-dependencies" in metadata
        assert expected_specifier in metadata["optional-dependencies"]["dev"]

    def test_preserves_non_workspace_optional_dependency(self, workspace: Path) -> None:
        hook = create_hook(workspace / "main", {"strategy": "allow-all-updates"})
        metadata = {"name": "main"}

        hook.update(metadata)

        assert "optional-dependencies" in metadata
        assert "pytest>=7.0" in metadata["optional-dependencies"]["dev"]

    @pytest.mark.parametrize(
        ("strategy", "expected_specifier"),
        [
            ("pin", "dep-with-extras[extra1,extra2]==4.0.0"),
            ("allow-patch-updates", "dep-with-extras[extra1,extra2]<4.1.0,>=4.0.0"),
            ("allow-minor-updates", "dep-with-extras[extra1,extra2]<5.0.0,>=4.0.0"),
            ("allow-all-updates", "dep-with-extras[extra1,extra2]>=4.0.0"),
            ("semver", "dep-with-extras[extra1,extra2]<5.0.0,>=4.0.0"),
        ],
    )
    def test_preserves_extras(self, workspace: Path, strategy: str, expected_specifier: str) -> None:
        hook = create_hook(workspace / "main", {"strategy": strategy})
        metadata = {"name": "main"}

        hook.update(metadata)

        assert expected_specifier in metadata["dependencies"]

    def test_does_not_modify_metadata_when_no_dependencies(self, workspace: Path) -> None:
        pkg = workspace / "no-deps"
        pkg.mkdir()
        (pkg / "pyproject.toml").write_text('[project]\nname = "no-deps"\nversion = "1.0.0"\n')

        hook = create_hook(pkg, {"strategy": "allow-all-updates"})
        metadata = {"name": "no-deps"}

        hook.update(metadata)

        assert metadata == {"name": "no-deps"}

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

    def test_override_dependencies(self, workspace: Path) -> None:
        hook = create_hook(workspace / "main", {"strategy": "allow-all-updates", "overrides": {"dep": "pin"}})
        metadata = {"name": "main"}

        hook.update(metadata)

        assert "dep==2.0.0" in metadata["dependencies"]
        assert "dep-with-extras[extra1,extra2]>=4.0.0" in metadata["dependencies"]

    def test_override_invalid_strategy_raises(self, workspace: Path) -> None:
        hook = create_hook(workspace / "main", {"strategy": "allow-all-updates", "overrides": {"dep": "invalid"}})
        metadata = {"name": "main"}

        with pytest.raises(ValueError, match="Invalid strategy 'invalid'"):
            hook.update(metadata)
