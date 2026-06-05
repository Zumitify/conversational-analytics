"""Application configuration loaded from config/app.yaml + environment.

Environment overrides use the ``CAE_`` prefix, e.g. ``CAE_DB_PATH``,
``CAE_LLM_PROVIDER``, ``CAE_LLM_MODEL``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class LimitsConfig(BaseModel):
    default_limit: int = 1000        # injected when the user didn't ask
    max_rows: int = 10_000           # hard cap enforced by the validator
    lookback_years: int = 5          # mandatory time bound when none given
    query_timeout_s: int = 30
    max_llm_retries: int = 2


class LLMConfig(BaseModel):
    provider: str = "anthropic"      # anthropic | mock
    model: str = "claude-opus-4-8"
    max_tokens: int = 2048
    summary_max_tokens: int = 512


class DatabaseConfig(BaseModel):
    dialect: str = "duckdb"          # duckdb | postgres
    path: str = "cae.duckdb"         # duckdb file
    postgres_dsn: str | None = None


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(BaseModel):
    semantic_layer_path: str = "config/semantic_layer.yaml"
    sessions_db_path: str = "sessions.db"
    database: DatabaseConfig = DatabaseConfig()
    llm: LLMConfig = LLMConfig()
    limits: LimitsConfig = LimitsConfig()
    api: APIConfig = APIConfig()


_ENV_OVERRIDES = {
    "CAE_DB_PATH": ("database", "path"),
    "CAE_DB_DIALECT": ("database", "dialect"),
    "CAE_POSTGRES_DSN": ("database", "postgres_dsn"),
    "CAE_LLM_PROVIDER": ("llm", "provider"),
    "CAE_LLM_MODEL": ("llm", "model"),
    "CAE_SEMANTIC_LAYER": ("semantic_layer_path",),
    "CAE_SESSIONS_DB": ("sessions_db_path",),
}


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from YAML (if present), then apply CAE_* env overrides."""
    raw: dict = {}
    candidate = Path(path) if path else Path("config/app.yaml")
    if candidate.exists():
        raw = yaml.safe_load(candidate.read_text()) or {}
    config = AppConfig.model_validate(raw)

    for env, keys in _ENV_OVERRIDES.items():
        value = os.environ.get(env)
        if value is None:
            continue
        target: object = config
        for key in keys[:-1]:
            target = getattr(target, key)
        setattr(target, keys[-1], value)
    return config
