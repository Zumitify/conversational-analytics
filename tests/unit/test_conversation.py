"""Session store: persistence, slot memory, follow-up context."""

from __future__ import annotations

from cae.conversation import SessionStore
from cae.models import (
    QueryIntent,
    QueryPlan,
    ResultDigest,
    SelectItem,
    Turn,
)


def make_turn(question: str, metrics: list[str]) -> Turn:
    intent = QueryIntent(metrics=metrics)
    return Turn(
        user_text=question,
        intent=intent,
        plan=QueryPlan(
            select=[SelectItem(alias=metrics[0], expr="SUM(1)", role="metric")],
            from_table="orders",
            from_physical="orders",
        ),
        sql="SELECT 1",
        result_digest=ResultDigest(columns=[metrics[0]], row_count=1),
    )


class TestSessionStore:
    def test_create_and_get(self, tmp_path):
        store = SessionStore(str(tmp_path / "s.db"))
        session = store.create_session()
        loaded = store.get_session(session.session_id)
        assert loaded is not None
        assert loaded.turns == []
        assert loaded.active_intent is None

    def test_unknown_session_returns_none(self, tmp_path):
        store = SessionStore(str(tmp_path / "s.db"))
        assert store.get_session("nope") is None

    def test_append_turn_promotes_active_intent(self, tmp_path):
        store = SessionStore(str(tmp_path / "s.db"))
        session = store.create_session()
        store.append_turn(session.session_id, make_turn("q1", ["revenue"]))
        loaded = store.get_session(session.session_id)
        assert len(loaded.turns) == 1
        assert loaded.active_intent.metrics == ["revenue"]

    def test_turns_ordered(self, tmp_path):
        store = SessionStore(str(tmp_path / "s.db"))
        session = store.create_session()
        store.append_turn(session.session_id, make_turn("first", ["revenue"]))
        store.append_turn(session.session_id, make_turn("second", ["units_sold"]))
        loaded = store.get_session(session.session_id)
        assert [t.user_text for t in loaded.turns] == ["first", "second"]
        assert loaded.active_intent.metrics == ["units_sold"]

    def test_survives_reopen(self, tmp_path):
        path = str(tmp_path / "s.db")
        store = SessionStore(path)
        session = store.create_session()
        store.append_turn(session.session_id, make_turn("persisted", ["revenue"]))
        store.close()

        reopened = SessionStore(path)
        loaded = reopened.get_session(session.session_id)
        assert loaded.turns[0].user_text == "persisted"

    def test_list_sessions(self, tmp_path):
        store = SessionStore(str(tmp_path / "s.db"))
        a = store.create_session()
        b = store.create_session()
        assert set(store.list_sessions()) == {a.session_id, b.session_id}
