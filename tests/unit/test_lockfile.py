import textwrap
from pathlib import Path

import pytest
from packaging.version import Version

from hatch_cada.lockfile import Lockfile, Package


class TestLockfile:
    def test_load_from_path(self, tmp_path: Path) -> None:
        lock_content = textwrap.dedent("""
            version = 1
            requires-python = ">=3.12"

            [[package]]
            name = "foo"
            version = "1.0.0"
        """)
        lock_path = tmp_path / "uv.lock"
        lock_path.write_text(lock_content)

        lockfile = Lockfile.load(lock_path)

        assert lockfile._root == tmp_path


class TestGetPackage:
    def test_returns_package_with_version(self, tmp_path: Path) -> None:
        lock_content = textwrap.dedent("""
            version = 1

            [[package]]
            name = "requests"
            version = "2.31.0"
        """)
        lock_path = tmp_path / "uv.lock"
        lock_path.write_text(lock_content)

        lockfile = Lockfile.load(lock_path)
        pkg = lockfile.get_package("requests")

        assert pkg == Package(name="requests", version=Version("2.31.0"), editable_path=None)

    def test_returns_package_with_editable_path(self, tmp_path: Path) -> None:
        pkg_dir = tmp_path / "packages" / "mylib"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "pyproject.toml").write_text(
            textwrap.dedent("""
                [project]
                name = "mylib"
                version = "3.5.0"
            """)
        )

        lock_content = textwrap.dedent("""
            version = 1

            [[package]]
            name = "mylib"
            source = { editable = "packages/mylib" }
        """)
        lock_path = tmp_path / "uv.lock"
        lock_path.write_text(lock_content)

        lockfile = Lockfile.load(lock_path)
        pkg = lockfile.get_package("mylib")

        assert pkg == Package(name="mylib", version=Version("3.5.0"), editable_path="packages/mylib")

    def test_raises_for_missing_package(self, tmp_path: Path) -> None:
        lock_content = textwrap.dedent("""
            version = 1

            [[package]]
            name = "foo"
            version = "1.0.0"
        """)
        lock_path = tmp_path / "uv.lock"
        lock_path.write_text(lock_content)

        lockfile = Lockfile.load(lock_path)

        with pytest.raises(KeyError, match="Package 'nonexistent' not found"):
            lockfile.get_package("nonexistent")

    def test_raises_for_package_without_version_or_editable(self, tmp_path: Path) -> None:
        lock_content = textwrap.dedent("""
            version = 1

            [[package]]
            name = "broken"
            source = { registry = "https://pypi.org/simple" }
        """)
        lock_path = tmp_path / "uv.lock"
        lock_path.write_text(lock_content)

        lockfile = Lockfile.load(lock_path)

        with pytest.raises(KeyError, match="has no version and is not editable"):
            lockfile.get_package("broken")
