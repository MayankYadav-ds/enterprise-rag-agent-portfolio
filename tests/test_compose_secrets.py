"""Production Compose grants each Qdrant API-key copy only to its consumer."""

from pathlib import Path

import yaml


def test_qdrant_api_key_secret_sources_are_distinct_and_service_scoped():
    compose_path = Path(__file__).resolve().parents[1] / "compose.production.yaml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    api_key_secret = compose["secrets"]["api_qdrant_api_key"]["file"]
    qdrant_key_secret = compose["secrets"]["qdrant_server_api_key"]["file"]
    assert api_key_secret.startswith("${QDRANT_API_KEY_FILE:")
    assert qdrant_key_secret.startswith("${QDRANT_SERVER_API_KEY_FILE:")
    assert api_key_secret != qdrant_key_secret

    api_secrets = {item["source"] for item in compose["services"]["api"]["secrets"]}
    qdrant_secrets = {item["source"] for item in compose["services"]["qdrant"]["secrets"]}
    assert "api_qdrant_api_key" in api_secrets
    assert "qdrant_server_api_key" not in api_secrets
    assert "qdrant_server_api_key" in qdrant_secrets
    assert "api_qdrant_api_key" not in qdrant_secrets
    assert compose["services"]["api"]["environment"]["QDRANT_API_KEY_FILE"] == (
        "/run/secrets/qdrant_api_key"
    )
