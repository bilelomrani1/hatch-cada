import subprocess
import textwrap
from pathlib import Path

from packaging.version import Version

from hatch_cada.pyproject import Pyproject


class TestHatchVcsIntegration:
    def test_resolves_version_from_git_tag(self, git_repo: Path) -> None:
        pyproject_path = git_repo / "pyproject.toml"
        pyproject_path.write_text(
            textwrap.dedent("""
                [project]
                name = "foo"
                dynamic = ["version"]

                [tool.hatch.version]
                source = "vcs"
            """)
        )
        subprocess.run(["git", "add", "."], cwd=git_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=git_repo, check=True, capture_output=True)
        subprocess.run(["git", "tag", "1.2.3"], cwd=git_repo, check=True, capture_output=True)

        pyproject = Pyproject.load(pyproject_path)

        assert pyproject.version == Version("1.2.3")

    def test_resolves_version_with_v_prefix_tag(self, git_repo: Path) -> None:
        pyproject_path = git_repo / "pyproject.toml"
        pyproject_path.write_text(
            textwrap.dedent("""
                [project]
                name = "foo"
                dynamic = ["version"]

                [tool.hatch.version]
                source = "vcs"
            """)
        )
        subprocess.run(["git", "add", "."], cwd=git_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=git_repo, check=True, capture_output=True)
        subprocess.run(["git", "tag", "v2.0.0"], cwd=git_repo, check=True, capture_output=True)

        pyproject = Pyproject.load(pyproject_path)

        assert pyproject.version == Version("2.0.0")

    def test_generates_dev_version_when_no_tags(self, git_repo: Path) -> None:
        pyproject_path = git_repo / "pyproject.toml"
        pyproject_path.write_text(
            textwrap.dedent("""
                [project]
                name = "foo"
                dynamic = ["version"]

                [tool.hatch.version]
                source = "vcs"
            """)
        )
        subprocess.run(["git", "add", "."], cwd=git_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=git_repo, check=True, capture_output=True)

        pyproject = Pyproject.load(pyproject_path)

        assert pyproject.version.is_devrelease
