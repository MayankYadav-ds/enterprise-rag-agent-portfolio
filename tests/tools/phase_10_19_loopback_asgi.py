"""TEST-ONLY loopback ASGI adapter for Phase 10.19; never a production server."""

from __future__ import annotations

import asyncio
import importlib.util
import ipaddress
import json
import os
import socket
import sys
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlsplit

MAX_BODY = 22 * 1024 * 1024
MAX_HEADERS = 64 * 1024


def disable_outbound_network() -> None:
    """Fail closed if application code attempts external networking."""
    original_getaddrinfo = socket.getaddrinfo

    def loopback_only_getaddrinfo(host, *args, **kwargs):
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if host != "localhost":
                raise OSError("External name resolution is disabled in this test harness") from None
            return original_getaddrinfo(host, *args, **kwargs)
        if not address.is_loopback:
            raise OSError("External name resolution is disabled in this test harness")
        return original_getaddrinfo(host, *args, **kwargs)

    def deny_connect(self, _address):
        raise OSError("Outbound network connections are disabled in this test harness")

    socket.getaddrinfo = loopback_only_getaddrinfo
    socket.socket.connect = deny_connect
    socket.socket.connect_ex = deny_connect


def load_demo_module(repo: Path):
    path = repo / "tests" / "test_phase_10_15_offline_pdf_demo.py"
    spec = importlib.util.spec_from_file_location("phase1019_synthetic_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the existing synthetic fixture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def read_body(reader: asyncio.StreamReader, headers: list[tuple[bytes, bytes]]) -> bytes:
    transfer = next(
        (v.decode("latin-1").lower() for k, v in headers if k == b"transfer-encoding"), ""
    )
    if "chunked" in transfer:
        result = bytearray()
        while True:
            line = await asyncio.wait_for(reader.readline(), 15)
            size = int(line.split(b";", 1)[0].strip(), 16)
            if size == 0:
                while await asyncio.wait_for(reader.readline(), 15) not in (b"\r\n", b"\n", b""):
                    pass
                break
            if len(result) + size > MAX_BODY:
                raise ValueError("request body exceeds local harness bound")
            result.extend(await asyncio.wait_for(reader.readexactly(size), 15))
            await asyncio.wait_for(reader.readexactly(2), 15)
        return bytes(result)
    raw_length = next((v for k, v in headers if k == b"content-length"), b"0")
    length = int(raw_length or b"0")
    if length < 0 or length > MAX_BODY:
        raise ValueError("request body exceeds local harness bound")
    return await asyncio.wait_for(reader.readexactly(length), 15) if length else b""


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, app) -> None:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 15)
        if len(head) > MAX_HEADERS:
            raise ValueError("request headers exceed local harness bound")
        lines = head[:-4].split(b"\r\n")
        method_raw, target_raw, version_raw = lines[0].split(b" ", 2)
        headers = []
        for line in lines[1:]:
            if b":" not in line:
                raise ValueError("malformed HTTP header")
            name, value = line.split(b":", 1)
            headers.append((name.strip().lower(), value.strip()))
        body = await read_body(reader, headers)
        target = urlsplit(target_raw.decode("ascii"))
        client = writer.get_extra_info("peername") or ("127.0.0.1", 0)
        server = writer.get_extra_info("sockname") or ("127.0.0.1", 8765)
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": version_raw.decode("ascii").split("/", 1)[-1],
            "method": method_raw.decode("ascii"),
            "scheme": "http",
            "path": target.path,
            "raw_path": target.path.encode("ascii"),
            "query_string": target.query.encode("ascii"),
            "root_path": "",
            "headers": headers,
            "client": (client[0], client[1]),
            "server": (server[0], server[1]),
            "state": {},
        }
        sent_body = False
        response_status = 500
        response_headers: list[tuple[bytes, bytes]] = []
        response_parts: list[bytes] = []

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal response_status, response_headers
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                response_parts.append(message.get("body", b""))

        await asyncio.wait_for(app(scope, receive, send), 90)
        payload = b"".join(response_parts)
        if not any(name.lower() == b"content-length" for name, _ in response_headers):
            response_headers.append((b"content-length", str(len(payload)).encode("ascii")))
        response_headers = [
            (name, value)
            for name, value in response_headers
            if name.lower() not in {b"connection", b"transfer-encoding"}
        ]
        response_headers.append((b"connection", b"close"))
        try:
            phrase = HTTPStatus(response_status).phrase
        except ValueError:
            phrase = "Unknown"
        writer.write(f"HTTP/1.1 {response_status} {phrase}\r\n".encode("ascii"))
        for name, value in response_headers:
            writer.write(name + b": " + value + b"\r\n")
        writer.write(b"\r\n" + payload)
        await writer.drain()
    except Exception as exc:
        # Never serialize request bodies, headers, credentials, or document text.
        detail = type(exc).__name__
        print(json.dumps({"event": "test_harness_error", "exception_type": detail}), flush=True)
        try:
            writer.write(
                b"HTTP/1.1 500 Internal Server Error\r\n"
                b"Content-Type: application/json\r\nContent-Length: 25\r\nConnection: close\r\n\r\n"
                b'{"detail":"test harness"}'
            )
            await writer.drain()
        except Exception:
            pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main(repo: Path, data_root: Path, host: str, port: int) -> None:
    if host != "127.0.0.1":
        raise SystemExit("Only the approved loopback bind address is allowed")
    disable_outbound_network()
    if data_root.exists():
        raise SystemExit("Refusing to reuse an existing synthetic runtime directory")
    data_root.mkdir(parents=True, mode=0o700)
    demo = load_demo_module(repo)
    import tiktoken

    from rag_agent.api.app import create_app
    from rag_agent.api.dependencies import get_generator_factory
    from rag_agent.api.documents import DocumentManager
    from rag_agent.api.settings import APISettings
    from rag_agent.auth.models import create_user
    from rag_agent.auth.password import hash_password
    from rag_agent.embeddings.fake import FakeEmbedder
    from rag_agent.generation import AnswerGenerator

    password = os.environ.get("PHASE1019_SYNTHETIC_PASSWORD")
    if not password or len(password) < 24:
        raise SystemExit("Missing ephemeral synthetic password")
    settings = APISettings(
        document_dir=data_root / "managed",
        qdrant_url="http://127.0.0.1:1",
        collection="phase1019_offline_only",
        environment="test",
        trusted_origins=("http://127.0.0.1:5175",),
        upload_rate_limit_attempts=50,
    )
    app = create_app(settings)
    runtime = app.state.runtime
    store = demo._MemoryVectorStore()
    embedder = demo._DemoEmbedder()
    runtime._store = lambda _embedder, *, validate: store
    runtime.embedding = lambda: embedder

    def forbidden_qdrant(*_args, **_kwargs):
        raise RuntimeError("real Qdrant construction is disabled in this harness")

    import rag_agent.api.runtime as runtime_module

    runtime_module._new_qdrant_client = forbidden_qdrant
    tiktoken.get_encoding = lambda _name: demo._WordTokenizer()

    def erase_vectors(manager: DocumentManager, doc_id: str, tenant_id: str) -> None:
        manager.runtime._store(FakeEmbedder(), validate=False).delete_document(
            doc_id, tenant_id=tenant_id
        )

    DocumentManager._erase_vectors = erase_vectors
    app.dependency_overrides[get_generator_factory] = lambda: (
        lambda: AnswerGenerator(demo._DeterministicEvidenceProvider())
    )
    async with app.router.lifespan_context(app):
        password_hash = hash_password(password)
        with runtime.connection() as connection:
            create_user(connection, "phase19-owner", password_hash, is_admin=True)
            create_user(connection, "phase19-tenant-b", password_hash, is_admin=False)
        legacy_vector = embedder.embed_query("93 kits had been delivered by 31 July 2026")
        store.points.append(
            {
                "chunk_id": "phase19-legacy-unscoped",
                "doc_id": "doc_legacy_unscoped",
                "source_file": "legacy-unscoped.pdf",
                "page_start": 1,
                "page_end": 1,
                "section_heading": None,
                "subheading": None,
                "chunk_type": "text",
                "text": "LEGACY_UNSCOPED_SENTINEL: 93 kits had been delivered.",
                "tenant_id": None,
                "vector": legacy_vector,
            }
        )
        server = await asyncio.start_server(
            lambda reader, writer: handle_client(reader, writer, app),
            host=host,
            port=port,
            limit=MAX_HEADERS,
        )
        print(
            json.dumps(
                {
                    "event": "test_harness_ready",
                    "host": host,
                    "port": port,
                    "adapter": "in-memory deterministic test-only",
                    "qdrant_client": "disabled",
                    "generation_provider": "deterministic test-only",
                    "model_loading": "disabled",
                }
            ),
            flush=True,
        )
        async with server:
            await server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: harness.py REPO DATA_DIR HOST PORT")
    asyncio.run(
        main(
            Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3], int(sys.argv[4])
        )
    )
