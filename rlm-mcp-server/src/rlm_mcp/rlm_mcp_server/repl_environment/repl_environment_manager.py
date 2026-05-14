from __future__ import annotations

from abc import ABC
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO


@dataclass
class REPLResponse:
    stdout: str
    stderr: str
    returncode: int | None = None

    def __str__(self) -> str:
        content = self.stdout if self.stdout else self.stderr
        return self._format_message(content)

    def emit(self) -> None:
        """Write captured output to the matching stream and include returncode."""
        if self.stdout:
            self._print_to_stream(self.stdout, stream=sys.stdout)
        if self.stderr:
            self._print_to_stream(self.stderr, stream=sys.stderr)

        rc_stream = sys.stderr if self.stderr else sys.stdout
        print()
        print(f"[return code={self.returncode}]", file=rc_stream)

    def _format_message(self, content: str) -> str:
        if not content:
            return f"[returncode={self.returncode}]"
        stripped = content.rstrip("\n")
        return f"{stripped}\n[returncode={self.returncode}]"

    @staticmethod
    def _print_to_stream(content: str, stream: TextIO) -> None:
        if content.endswith("\n"):
            print(content, end="", file=stream)
            return
        print(content, file=stream)


class BaseREPLEnvironmentManager(ABC):
    """
    Shared REPL lifecycle/IO logic used by different runtime backends.
    """

    def __init__(
        self,
        startup_timeout: float = 60.0,
        execution_timeout: float = 15,
    ) -> None:
        self.startup_timeout = startup_timeout
        self.execution_timeout = execution_timeout

        self._proc: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stderr_queue: queue.Queue[str] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._marker_prefix = "__REPL_DONE__"

    def _start_process(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        startup_error_context: str = "REPL",
    ) -> None:
        if self._proc is not None:
            raise RuntimeError("REPL is already running.")
        self._reset_runtime_state()

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=cwd,
        )

        assert self._proc.stdout is not None
        assert self._proc.stderr is not None

        self._stdout_thread = threading.Thread(
            target=self._reader,
            args=(self._proc.stdout, self._stdout_queue),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._reader,
            args=(self._proc.stderr, self._stderr_queue),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                stderr = self._collect_now(self._stderr_queue)
                raise RuntimeError(
                    f"{startup_error_context} exited during startup.\nstderr:\n{stderr}"
                )

            probe = self.execute("print('READY')", timeout=2.0, _startup_probe=True)
            if "READY" in probe.stdout:
                return

            time.sleep(0.1)

        self.stop()
        raise TimeoutError("Timed out waiting for REPL to start.")

    def start(self) -> None:
        raise NotImplementedError

    def execute(
        self,
        code: str,
        timeout: float | None = None,
        _startup_probe: bool = False,
    ) -> REPLResponse:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("REPL is not running. Call start() first.")
        if self._proc.poll() is not None:
            raise RuntimeError("REPL process has already exited.")

        timeout = timeout if timeout is not None else self.execution_timeout
        marker = f"{self._marker_prefix}_{uuid.uuid4().hex}"
        wrapped = self._wrap_code(code, marker)

        self._proc.stdin.write(wrapped)
        self._proc.stdin.flush()

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        deadline = time.time() + timeout
        found_marker = False

        while time.time() < deadline:
            self._drain_queue(self._stdout_queue, stdout_chunks)
            self._drain_queue(self._stderr_queue, stderr_chunks)

            combined_stdout = "".join(stdout_chunks)
            if marker in combined_stdout:
                found_marker = True
                break

            if self._proc.poll() is not None:
                break

            time.sleep(0.01)

        self._drain_queue(self._stdout_queue, stdout_chunks)
        self._drain_queue(self._stderr_queue, stderr_chunks)

        stdout = "".join(stdout_chunks)
        stderr = self._strip_repl_prompts("".join(stderr_chunks))

        if found_marker:
            stdout = (
                stdout
                .replace(marker + "\r\n", "")
                .replace(marker + "\n", "")
                .replace(marker, "")
            )
        elif not _startup_probe:
            raise TimeoutError(f"Execution timed out after {timeout} seconds.")

        return REPLResponse(
            stdout=stdout,
            stderr=stderr,
            returncode=self._proc.poll(),
        )

    def stop(self) -> None:
        try:
            if self._proc is not None:
                if self._proc.stdin:
                    try:
                        self._proc.stdin.write("exit()\n")
                        self._proc.stdin.flush()
                    except Exception:
                        pass

                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2)
                except Exception:
                    try:
                        self._proc.kill()
                        self._proc.wait(timeout=2)
                    except Exception:
                        pass
        finally:
            self._reset_runtime_state()

    def restart(self) -> None:
        self.stop()
        self.start()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def install_package(self, package: str, timeout: float = 60.0) -> REPLResponse:
        """Install a package inside the running REPL environment."""
        return self.execute(
            f"import subprocess; "
            f"subprocess.run(['python', '-m', 'pip', 'install', {package!r}], check=True)",
            timeout=timeout,
        )

    @staticmethod
    def _reader(stream: TextIO, output_queue: queue.Queue[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                output_queue.put(line)
        finally:
            stream.close()

    def _reset_runtime_state(self) -> None:
        self._proc = None
        self._stdout_queue = queue.Queue()
        self._stderr_queue = queue.Queue()
        self._stdout_thread = None
        self._stderr_thread = None


    @staticmethod
    def _drain_queue(q: queue.Queue[str], chunks: list[str]) -> None:
        while True:
            try:
                chunks.append(q.get_nowait())
            except queue.Empty:
                break

    @staticmethod
    def _collect_now(q: queue.Queue[str]) -> str:
        parts: list[str] = []
        while True:
            try:
                parts.append(q.get_nowait())
            except queue.Empty:
                return "".join(parts)

    @staticmethod
    def _strip_repl_prompts(stderr: str) -> str:
        # Interactive Python writes prompts (>>> / ...) to stderr; remove them from captured errors.
        return re.sub(r"^(?:>>> |\.\.\. )+", "", stderr, flags=re.MULTILINE)

    @staticmethod
    def _wrap_code(code: str, marker: str) -> str:
        # Run the user code from a repr-built source string so arbitrary quotes,
        # backslashes, Unicode, and newlines cannot break the wrapper itself.
        # Emit the completion marker through sys.stdout with an explicit flush so
        # execute() can reliably detect command completion, including on Windows.
        script = (
            "import sys, traceback\n"
            "try:\n"
            f"    exec(compile({code!r}, '<repl-input>', 'exec'), globals(), globals())\n"
            "except Exception:\n"
            "    traceback.print_exc()\n"
            f"sys.stdout.write({(marker + chr(10))!r})\n"
            "sys.stdout.flush()\n"
        )
        return f"exec({script!r}, globals(), globals())\n"

    def __enter__(self) -> "BaseREPLEnvironmentManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class DockerREPLEnvironmentManager(BaseREPLEnvironmentManager):
    def __init__(
        self,
        image: str = "python:3.12-slim",
        container_name: str | None = None,
        startup_timeout: float = 60.0,
        execution_timeout: float = 5.0,
        memory_limit: str | None = "256m",
        cpus: str | None = "1.0",
        network_disabled: bool = True,
        working_dir: str = "/workspace",
        extra_docker_args: list[str] | None = None,
    ) -> None:
        super().__init__(startup_timeout=startup_timeout, execution_timeout=execution_timeout)
        self.image = image
        self._object_id = uuid.uuid4().hex[:8]
        container_prefix = container_name or "repl-env"
        self._container_prefix = f"{container_prefix}-{self._object_id}"
        self.memory_limit = memory_limit
        self.cpus = cpus
        self.network_disabled = network_disabled
        self.working_dir = working_dir
        self.extra_docker_args = extra_docker_args or []
        self._session_id = 0
        self._active_container_name: str | None = None

    @property
    def container_name(self) -> str:
        return self._active_container_name or self._build_container_name(self._session_id + 1)

    def start(self) -> None:
        self._validate_extra_docker_args(self.extra_docker_args)

        self._session_id += 1
        self._active_container_name = self._build_container_name(self._session_id)

        docker_cmd = self._get_docker_base_cmd()
        docker_cwd = self._get_docker_cwd(docker_cmd)
        self._pull_image_if_needed(docker_cmd, cwd=docker_cwd)

        cmd = docker_cmd + [
            "run",
            "--rm",
            "-i",
            "--name",
            self.container_name,
            "--workdir",
            self.working_dir,
        ]

        if self.network_disabled:
            cmd.extend(["--network", "none"])
        if self.memory_limit:
            cmd.extend(["--memory", self.memory_limit])
        if self.cpus:
            cmd.extend(["--cpus", self.cpus])

        cmd.extend(self.extra_docker_args)
        cmd.extend([self.image, "python", "-u", "-i", "-q"])

        self._start_process(
            cmd,
            cwd=docker_cwd,
            startup_error_context="Container REPL",
        )

    def stop(self) -> None:
        container_name = self._active_container_name
        try:
            super().stop()
        finally:
            self._active_container_name = None

        try:
            docker_cmd = self._get_docker_base_cmd()
        except RuntimeError:
            docker_cmd = None

        if docker_cmd and container_name:
            subprocess.run(
                docker_cmd + ["rm", "-f", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=self._get_docker_cwd(docker_cmd),
            )

    def _pull_image_if_needed(self, docker_cmd: list[str], cwd: str | None = None) -> None:
        inspect = subprocess.run(
            docker_cmd + ["image", "inspect", self.image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
        )
        if inspect.returncode == 0:
            return

        pull = subprocess.run(
            docker_cmd + ["pull", self.image],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if pull.returncode != 0:
            raise RuntimeError(f"Failed to pull Docker image {self.image!r}.\n{pull.stderr}")

    def _build_container_name(self, session_id: int) -> str:
        return f"{self._container_prefix}-s{session_id}"

    @staticmethod
    def _validate_extra_docker_args(args: list[str]) -> None:
        disallowed = ("-v", "--volume", "--mount")
        for arg in args:
            lower = arg.lower()
            if (
                lower in disallowed
                or lower.startswith("-v")
                or lower.startswith("--volume=")
                or lower.startswith("--mount=")
            ):
                raise ValueError(
                    "Mount options are not allowed in extra_docker_args. "
                    "Host file access must remain isolated."
                )

    @staticmethod
    def _docker_command_works(
        docker_cmd: list[str],
        cwd: str | None = None,
    ) -> bool:
        try:
            check = subprocess.run(
                docker_cmd + ["version", "--format", "{{.Server.Version}}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
                cwd=cwd,
            )
            return check.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _get_docker_base_cmd(self) -> list[str]:
        native_docker = shutil.which("docker")
        if native_docker and self._docker_command_works(
            [native_docker],
            cwd=self._get_docker_cwd([native_docker]),
        ):
            return [native_docker]

        wsl_path = shutil.which("wsl")
        if wsl_path and self._docker_command_works(
            [wsl_path, "docker"],
            cwd=self._get_docker_cwd([wsl_path, "docker"]),
        ):
            return [wsl_path, "docker"]

        details = []
        if native_docker:
            details.append("Found 'docker' but it is not usable")
        if wsl_path:
            details.append("Found 'wsl' but 'wsl docker' is not usable")

        suffix = f" ({'; '.join(details)})" if details else ""
        raise RuntimeError(
            "Could not find a usable Docker command. Install/enable Docker Desktop or Docker in WSL"
            f" and ensure the daemon is running{suffix}."
        )

    @staticmethod
    def _get_docker_cwd(docker_cmd: list[str]) -> str | None:
        """Use a stable cwd for wsl-backed Docker to avoid caller cwd translation issues."""
        if not docker_cmd:
            return None

        executable_name = Path(docker_cmd[0]).name.lower()
        if executable_name in {"wsl", "wsl.exe"}:
            return str(Path.home())

        return None


class SubprocessREPLEnvironmentManager(BaseREPLEnvironmentManager):
    def __init__(
        self,
        startup_timeout: float = 60.0,
        execution_timeout: float = 5.0,
        python_executable: str = sys.executable,
        working_dir: str | None = None,
    ) -> None:
        super().__init__(startup_timeout=startup_timeout, execution_timeout=execution_timeout)
        self.python_executable = python_executable
        self.working_dir = working_dir

    def start(self) -> None:
        cmd = [self.python_executable, "-u", "-i", "-q"]
        self._start_process(
            cmd,
            cwd=self.working_dir,
            startup_error_context="Subprocess REPL",
        )


class REPLEnvironmentManager:
    """
    Facade that exposes the same API while letting callers choose backend.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        container_name: str | None = None,
        startup_timeout: float = 60.0,
        execution_timeout: float = 5.0,
        memory_limit: str | None = "256m",
        cpus: str | None = "1.0",
        network_disabled: bool = True,
        working_dir: str = "/workspace",
        extra_docker_args: list[str] | None = None,
        *,
        environment_type: Literal["docker", "subprocess"] = "docker",
        python_executable: str = sys.executable,
        local_working_dir: str | None = None,
    ) -> None:
        self.environment_type = environment_type
        if environment_type == "docker":
            self._backend: BaseREPLEnvironmentManager = DockerREPLEnvironmentManager(
                image=image,
                container_name=container_name,
                startup_timeout=startup_timeout,
                execution_timeout=execution_timeout,
                memory_limit=memory_limit,
                cpus=cpus,
                network_disabled=network_disabled,
                working_dir=working_dir,
                extra_docker_args=extra_docker_args,
            )
            return

        if environment_type == "subprocess":
            self._backend = SubprocessREPLEnvironmentManager(
                startup_timeout=startup_timeout,
                execution_timeout=execution_timeout,
                python_executable=python_executable,
                working_dir=local_working_dir,
            )
            return

        raise ValueError(
            "Unsupported environment_type. Expected 'docker' or 'subprocess'."
        )

    @property
    def backend(self) -> BaseREPLEnvironmentManager:
        return self._backend

    @property
    def container_name(self) -> str | None:
        return getattr(self._backend, "container_name", None)

    def start(self) -> None:
        self._backend.start()

    def execute(
        self,
        code: str,
        timeout: float | None = None,
        _startup_probe: bool = False,
    ) -> REPLResponse:
        return self._backend.execute(code, timeout=timeout, _startup_probe=_startup_probe)

    def stop(self) -> None:
        self._backend.stop()

    def restart(self) -> None:
        self._backend.restart()

    def is_running(self) -> bool:
        return self._backend.is_running()

    def install_package(self, package: str, timeout: float = 60.0) -> REPLResponse:
        return self._backend.install_package(package, timeout=timeout)

    def __enter__(self) -> "REPLEnvironmentManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @staticmethod
    def _validate_extra_docker_args(args: list[str]) -> None:
        DockerREPLEnvironmentManager._validate_extra_docker_args(args)

    @staticmethod
    def _get_docker_cwd(docker_cmd: list[str]) -> str | None:
        return DockerREPLEnvironmentManager._get_docker_cwd(docker_cmd)

    @staticmethod
    def _strip_repl_prompts(stderr: str) -> str:
        return BaseREPLEnvironmentManager._strip_repl_prompts(stderr)

if __name__ == "__main__":

    print("TESTING SubprocessREPLEnvironmentManager")
    with REPLEnvironmentManager(
        image="python:3.12-slim",
        memory_limit="256m",
        cpus="1.0",
        network_disabled=True,
        environment_type="subprocess",
    ) as repl:
        print(repl.execute("x = 21"))
        print(repl.execute("print(x * 2)").stdout)   # 42
        print(repl.execute("import sys; print(sys.version)").stdout)

        err = repl.execute("raise ValueError('test')")
        print("STDERR:")
        print(err.stderr)

    print("TESTING DockerREPLEnvironmentManager")
    with REPLEnvironmentManager(
            image="python:3.12-slim",
            memory_limit="256m",
            cpus="1.0",
            network_disabled=True,
            environment_type="docker",
    ) as repl:
        print(repl.execute("x = 21"))
        print(repl.execute("print(x * 2)").stdout)   # 42
        print(repl.execute("import sys; print(sys.version)").stdout)

        err = repl.execute("raise ValueError('test')")
        print("STDERR:")
        print(err.stderr)
