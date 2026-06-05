"""Shared fixtures: semantic layer, seeded DuckDB, mock-LLM pipeline.

Everything is anchored to a frozen TODAY so relative time ranges resolve
deterministically.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from cae.config import AppConfig, DatabaseConfig
from cae.conversation import SessionStore
from cae.data.seed import seed_duckdb
from cae.execution import DuckDBEngine
from cae.llm.client import MockProvider
from cae.pipeline import Pipeline
from cae.semantic_layer import SemanticLayer

TODAY = date(2026, 6, 4)
REPO_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_LAYER_PATH = REPO_ROOT / "config" / "semantic_layer.yaml"


@pytest.fixture(scope="session")
def layer() -> SemanticLayer:
    return SemanticLayer.from_yaml(SEMANTIC_LAYER_PATH)


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "test.duckdb"
    seed_duckdb(path, n_orders=3000, n_customers=500, n_products=120,
                seed=7, end_date=TODAY)
    return path


@pytest.fixture(scope="session")
def engine(db_path) -> DuckDBEngine:
    eng = DuckDBEngine(str(db_path))
    yield eng
    eng.close()


@pytest.fixture()
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture()
def pipeline(db_path, layer, mock_provider, tmp_path) -> Pipeline:
    config = AppConfig(
        semantic_layer_path=str(SEMANTIC_LAYER_PATH),
        sessions_db_path=str(tmp_path / "sessions.db"),
        database=DatabaseConfig(dialect="duckdb", path=str(db_path)),
    )
    store = SessionStore(config.sessions_db_path)
    return Pipeline(
        config,
        provider=mock_provider,
        layer=layer,
        store=store,
        today=TODAY,
    )
