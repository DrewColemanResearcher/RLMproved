from collections import OrderedDict
from threading import RLock
import atexit
import os
from typing import Literal, Optional

from rlm_mcp.rlm_mcp_server.repl_environment.repl_environment_manager import REPLEnvironmentManager


class SessionManager:
    # TODO find out why on multithreading it deletes the context variable
    _instance = None

    def __init__(self, max_sessions: int = 20, environment_type: Optional[str] = None):
        if SessionManager._instance is not None:
            raise Exception("This class is a singleton!")
        self.max_sessions = max_sessions
        self.environment_type: Literal["docker", "subprocess"] = self._resolve_environment_type(
            environment_type or os.getenv("RLM_REPL_ENVIRONMENT", "docker")
        )
        self.sessions: OrderedDict[tuple[str, str], REPLEnvironmentManager] = OrderedDict()
        self.lock = RLock()

    @staticmethod
    def _resolve_environment_type(value: str) -> Literal["docker", "subprocess"]:
        return "subprocess" if value == "subprocess" else "docker"

    @classmethod
    def get_instance(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, user_id: str, chat_id: str) -> REPLEnvironmentManager:
        key = (user_id, chat_id)

        with self.lock:
            if key in self.sessions:
                session = self.sessions.pop(key)
                self.sessions[key] = session  # mark as most recently used
                return session

            session = REPLEnvironmentManager(environment_type=self.environment_type)
            session.start()

            self.sessions[key] = session
            self._evict_if_needed()
            return session

    def stop(self, user_id: str, chat_id: str) -> None:
        key = (user_id, chat_id)

        with self.lock:
            session = self.sessions.pop(key, None)

        if session is not None:
            try:
                session.stop()
            except Exception:
                pass

    def stop_all(self) -> None:
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()

        for session in sessions:
            try:
                session.stop()
            except Exception:
                pass

    def _evict_if_needed(self) -> None:
        while len(self.sessions) > self.max_sessions:
            _, old_session = self.sessions.popitem(last=False)  # LRU entry
            try:
                old_session.stop()
            except Exception:
                pass


session_manager = SessionManager.get_instance()
atexit.register(session_manager.stop_all)
