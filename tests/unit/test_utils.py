from pathlib import Path

import pytest

from hatch_cada.utils import find_workspace_root


class TestFindWorkspaceRoot:
    def test_returns_env_var_when_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        monkeypatch.setenv("WORKSPACE_ROOT", str(workspace_root))

        result = find_workspace_root(tmp_path)

        assert result == workspace_root

    def test_walks_up_to_find_uv_lock(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WORKSPACE_ROOT", raising=False)

        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        (workspace_root / "uv.lock").touch()

        nested_dir = workspace_root / "packages" / "foo"
        nested_dir.mkdir(parents=True)

        result = find_workspace_root(nested_dir)

        assert result == workspace_root

    def test_returns_none_when_no_workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WORKSPACE_ROOT", raising=False)

        result = find_workspace_root(tmp_path)

        assert result is None

    def test_env_var_takes_precedence_over_uv_lock(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_workspace = tmp_path / "env-workspace"
        env_workspace.mkdir()
        monkeypatch.setenv("WORKSPACE_ROOT", str(env_workspace))

        local_workspace = tmp_path / "local-workspace"
        local_workspace.mkdir()
        (local_workspace / "uv.lock").touch()

        result = find_workspace_root(local_workspace)

        assert result == env_workspace
