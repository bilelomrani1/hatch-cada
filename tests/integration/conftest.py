import os
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def git_repo(tmp_path: Path) -> Generator[Path, None, None]:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(prev_cwd)
