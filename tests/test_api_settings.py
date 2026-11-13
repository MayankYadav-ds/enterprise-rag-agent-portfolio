"""Configuration tests for production Qdrant transport and secret files."""

import os
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_agent.api.settings import APISettings


def test_qdrant_api_key_file_strips_only_trailing_newlines_and_is_hidden(tmp_path):
    secret_file = tmp_path / "qdrant-key"
    secret_file.write_bytes(b"  local-key  \r\n")
    if os.name == "posix":
        secret_file.chmod(0o600)
    settings = APISettings(
        qdrant_url="https://qdrant.example.com:6333",
        qdrant_api_key_file=secret_file,
    )
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == "  local-key  "
    assert "local-key" not in repr(settings) + settings.model_dump_json()


def test_inline_qdrant_key_takes_precedence_without_reading_secret_file(monkeypatch, tmp_path):
    missing_file = tmp_path / "missing-key"
    monkeypatch.setenv("RAG_ENV", "test")
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example.com:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "inline-value")
    monkeypatch.setenv("QDRANT_API_KEY_FILE", str(missing_file))
    settings = APISettings.from_env()
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == "inline-value"


def test_blank_inline_key_falls_back_to_secret_file(monkeypatch, tmp_path):
    secret_file = tmp_path / "qdrant-key"
    secret_file.write_text("file-value\n", encoding="utf-8")
    if os.name == "posix":
        secret_file.chmod(0o600)
    monkeypatch.setenv("RAG_ENV", "test")
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example.com:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "")
    monkeypatch.setenv("QDRANT_API_KEY_FILE", str(secret_file))
    settings = APISettings.from_env()
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == "file-value"


@pytest.mark.parametrize("content", [b"", b"\r\n", b"first\nsecond", b"key\x00value"])
def test_empty_or_multiline_qdrant_secret_is_rejected(tmp_path, content):
    secret_file = tmp_path / "qdrant-key"
    secret_file.write_bytes(content)
    if os.name == "posix":
        secret_file.chmod(0o600)
    with pytest.raises(ValidationError, match="QDRANT_API_KEY_FILE"):
        APISettings(qdrant_api_key_file=secret_file)


def test_missing_qdrant_secret_file_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="QDRANT_API_KEY_FILE"):
        APISettings(qdrant_api_key_file=tmp_path / "missing-key")


def test_symlink_qdrant_secret_file_is_rejected(monkeypatch, tmp_path):
    secret_file = tmp_path / "qdrant-key"
    secret_file.write_text("test-only-placeholder", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def is_symlink(path):
        if path == secret_file:
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(ValidationError, match="QDRANT_API_KEY_FILE"):
        APISettings(qdrant_api_key_file=secret_file)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are unavailable")
def test_permissive_qdrant_secret_file_is_rejected(tmp_path):
    secret_file = tmp_path / "qdrant-key"
    secret_file.write_text("secret", encoding="utf-8")
    secret_file.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IROTH)
    with pytest.raises(ValidationError, match="owner-only"):
        APISettings(qdrant_api_key_file=secret_file)


def test_ca_file_must_be_valid_and_readable(tmp_path):
    bad_ca = tmp_path / "ca.pem"
    bad_ca.write_text("not a certificate", encoding="utf-8")
    with pytest.raises(ValidationError, match="QDRANT_CA_CERT_FILE"):
        APISettings(
            qdrant_url="https://qdrant.example.com:6333",
            qdrant_api_key="key",
            qdrant_ca_cert_file=bad_ca,
        )


def test_unreadable_ca_file_is_rejected(tmp_path, monkeypatch):
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr("rag_agent.api.settings.os.access", lambda _path, _mode: False)
    with pytest.raises(ValidationError, match="QDRANT_CA_CERT_FILE"):
        APISettings(
            qdrant_url="https://qdrant.example.com:6333",
            qdrant_api_key="key",
            qdrant_ca_cert_file=ca_file,
        )
