"""Uvicorn-in-a-thread helpers shared by the run orchestrator and tests."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import uvicorn


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class ServerHandle:
    server: uvicorn.Server
    thread: threading.Thread
    port: int

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=timeout)


def start_uvicorn(app, *, port: int | None = None,
                  startup_timeout_s: float = 15.0) -> ServerHandle:
    port = port or free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + startup_timeout_s
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("uvicorn thread died during startup")
        if time.time() > deadline:
            raise RuntimeError(f"uvicorn did not start in {startup_timeout_s}s")
        time.sleep(0.01)
    return ServerHandle(server=server, thread=thread, port=port)
