from __future__ import annotations

import pickle
import socket
import struct
from typing import Any


HEADER_STRUCT = struct.Struct("!Q")


def dumps_pickle(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def loads_pickle(data: bytes) -> Any:
    return pickle.loads(data)


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_framed_bytes(sock: socket.socket, payload: bytes) -> int:
    frame = HEADER_STRUCT.pack(len(payload)) + payload
    sock.sendall(frame)
    return len(frame)


def recv_framed_bytes(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, HEADER_STRUCT.size)
    (size,) = HEADER_STRUCT.unpack(header)
    return _recv_exact(sock, int(size))


__all__ = [
    "dumps_pickle",
    "loads_pickle",
    "send_framed_bytes",
    "recv_framed_bytes",
]
