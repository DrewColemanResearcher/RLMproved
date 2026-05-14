from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

import sys

from rlm_mcp.rlm_mcp_server.repl_environment.repl_environment_manager import (
    DockerREPLEnvironmentManager,
    REPLEnvironmentManager,
    REPLResponse,
    SubprocessREPLEnvironmentManager,
)


def test_manager_objects_get_different_container_names() -> None:
    first = DockerREPLEnvironmentManager(container_name="my-repl")
    second = DockerREPLEnvironmentManager(container_name="my-repl")

    assert first.container_name != second.container_name
    assert first.container_name.endswith("-s1")
    assert second.container_name.endswith("-s1")

def test_mount_options_are_rejected() -> None:
    with pytest.raises(ValueError):
        DockerREPLEnvironmentManager._validate_extra_docker_args(["-v", "C:\\host:/workspace"])

    with pytest.raises(ValueError):
        DockerREPLEnvironmentManager._validate_extra_docker_args(["--mount=type=bind,src=C:\\host,dst=/workspace"])

def test_get_docker_base_cmd_prefers_native_docker() -> None:
    manager = DockerREPLEnvironmentManager()
    with patch("shutil.which", side_effect=["C:\\Program Files\\Docker\\docker.exe", "C:\\Windows\\System32\\wsl.exe"]):
        with patch.object(manager, "_docker_command_works", side_effect=[True, True]):
            assert manager._get_docker_base_cmd() == ["C:\\Program Files\\Docker\\docker.exe"]

def test_get_docker_base_cmd_falls_back_to_wsl() -> None:
    manager = DockerREPLEnvironmentManager()
    with patch("shutil.which", side_effect=["C:\\Program Files\\Docker\\docker.exe", "C:\\Windows\\System32\\wsl.exe"]):
        with patch.object(manager, "_docker_command_works", side_effect=[False, True]):
            assert manager._get_docker_base_cmd() == ["C:\\Windows\\System32\\wsl.exe", "docker"]

def test_stop_clears_active_container_name() -> None:
    manager = DockerREPLEnvironmentManager()
    manager._active_container_name = manager._build_container_name(1)

    with patch.object(manager, "_get_docker_base_cmd", return_value=["docker"]):
        with patch("subprocess.run"):
            manager.stop()

    assert manager._active_container_name is None


def test_strip_repl_prompts_from_stderr() -> None:
    stderr = ">>> >>> Traceback (most recent call last):\n... ValueError: boom\n"
    cleaned = DockerREPLEnvironmentManager._strip_repl_prompts(stderr)

    assert cleaned.startswith("Traceback")
    assert ">>>" not in cleaned


def test_get_docker_cwd_is_stable_for_wsl() -> None:
    cwd = DockerREPLEnvironmentManager._get_docker_cwd(["C:\\Windows\\System32\\wsl.exe", "docker"])
    assert isinstance(cwd, str)
    assert len(cwd) > 0


def test_get_docker_cwd_is_none_for_native_docker() -> None:
    assert DockerREPLEnvironmentManager._get_docker_cwd(["C:\\Program Files\\Docker\\docker.exe"]) is None


def test_facade_can_select_subprocess_backend() -> None:
    manager = REPLEnvironmentManager(environment_type="subprocess")

    assert isinstance(manager.backend, SubprocessREPLEnvironmentManager)
    assert manager.container_name is None


def test_subprocess_manager_persists_state_between_exec_calls() -> None:
    manager = SubprocessREPLEnvironmentManager(startup_timeout=10.0, execution_timeout=2.0)
    try:
        manager.start()
        manager.execute("x = 10")
        result = manager.execute("print(x * 2)")
        assert result.stderr == ""
        assert "20" in result.stdout
    finally:
        manager.stop()


def test_facade_rejects_unknown_environment_type() -> None:
    with pytest.raises(ValueError):
        REPLEnvironmentManager(environment_type=cast(object, "something-else"))


def test_repl_response_str_uses_stdout_and_returncode() -> None:
    response = REPLResponse(stdout="hello\n", stderr="", returncode=0)

    assert str(response) == "hello\n[returncode=0]"


def test_repl_response_emit_writes_stderr_and_returncode_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    response = REPLResponse(stdout="", stderr="boom\n", returncode=1)

    response.emit()
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == "boom\n[returncode=1]\n"


def test_repl_response_emit_writes_stdout_and_returncode_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    response = REPLResponse(stdout="ok\n", stderr="", returncode=0)

    response.emit()
    captured = capsys.readouterr()

    assert captured.out == "ok\n[returncode=0]\n"
    assert captured.err == ""

def test_works_with_leopardi():
    manager = REPLEnvironmentManager(environment_type="subprocess")
    doc = """
Sempre caro mi fu quest’ermo colle,
e questa siepe, che da tanta parte
dell’ultimo orizzonte il guardo esclude.
Ma sedendo e mirando, interminati
spazi di là da quella, e sovrumani
silenzi, e profondissima quïete
io nel pensier mi fingo, ove per poco
il cor non si spaura. E come il vento
odo stormir tra queste piante, io quello
infinito silenzio a questa voce
vo comparando: e mi sovvien l’eterno,
e le morte stagioni, e la presente
e viva, e il suon di lei. Così tra questa
immensità s’annega il pensier mio:
e il naufragar m’è dolce in questo mare.
    """.strip()
    code = f'context = {doc!r}'
    manager.start()
    manager.execute(code)
    manager.stop()

