"""Tenant identity validation shared by authenticated API boundaries."""

from __future__ import annotations

import re


def validate_tenant_id(value: str) -> str:
    """Accept only the canonical UUID-hex identity stored for authenticated users."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError("Invalid tenant identity")
    return value
