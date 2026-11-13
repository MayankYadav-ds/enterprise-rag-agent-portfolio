"""Forwarded client identity is accepted only from the configured Caddy peer."""

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

CADDY_IP = "172.30.105.2"


def _client_for_peer(peer: str) -> TestClient:
    app = FastAPI()

    @app.get("/peer")
    def peer_identity(request: Request) -> dict[str, str]:
        return {"host": request.client.host if request.client else "unknown"}

    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=CADDY_IP)
    return TestClient(app, client=(peer, 12345))


def test_trusted_caddy_forwarded_ip_is_used_for_rate_limit_identity():
    with _client_for_peer(CADDY_IP) as client:
        response = client.get("/peer", headers={"X-Forwarded-For": "198.51.100.40"})

    assert response.json() == {"host": "198.51.100.40"}


def test_untrusted_direct_client_cannot_spoof_forwarded_ip():
    with _client_for_peer("198.51.100.50") as client:
        response = client.get("/peer", headers={"X-Forwarded-For": "203.0.113.99"})

    assert response.json() == {"host": "198.51.100.50"}
