"""Streamlit UI: chat + table + chart + SQL inspector + editable intent.

The editable intent panel is the trust feature: when the model misreads a
question, the user fixes the structured intent directly and re-runs —
no prompt fighting.

Run with:  streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import json

import streamlit as st

from cae.config import load_config
from cae.exceptions import CAEError, ClarificationNeeded
from cae.models import QueryIntent
from cae.pipeline import Pipeline

st.set_page_config(page_title="Conversational Analytics", layout="wide")


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline(load_config())


pipeline = get_pipeline()

if "session_id" not in st.session_state:
    st.session_state.session_id = pipeline.create_session()
    st.session_state.history = []   # list[AskResponse]

st.title("Conversational Analytics Engine")
st.caption(
    f"session `{st.session_state.session_id}` — ask things like "
    "*\"weekly revenue trend in the Northeast this year, by product line\"*"
)

left, center = st.columns([1, 2])

# ---------------------------------------------------------------- chat pane
with left:
    st.subheader("Conversation")
    for response in st.session_state.history:
        with st.chat_message("user"):
            st.write(response.question)
        with st.chat_message("assistant"):
            st.write(response.summary or f"{response.result.row_count} rows.")

question = st.chat_input("Ask a question about the data…")
if question:
    try:
        with st.spinner("Thinking…"):
            response = pipeline.ask(st.session_state.session_id, question)
        st.session_state.history.append(response)
        st.rerun()
    except ClarificationNeeded as e:
        suffix = f" Did you mean: {', '.join(e.suggestions)}?" if e.suggestions else ""
        st.warning(f"{e.message}{suffix}")
    except CAEError as e:
        st.error(str(e))

# ------------------------------------------------------------- result pane
with center:
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.subheader("Result")

        chart = latest.chart_spec
        chart_type = chart.get("chart_type", "table")
        if chart_type == "kpi":
            st.metric(chart.get("metric", "value"), f"{chart.get('value'):,.2f}"
                      if isinstance(chart.get("value"), (int, float))
                      else str(chart.get("value")))
        elif "mark" in chart:
            st.vega_lite_chart(chart, use_container_width=True)

        columns = [c.name for c in latest.result.columns]
        st.dataframe(
            [dict(zip(columns, row)) for row in latest.result.rows],
            use_container_width=True,
        )
        if latest.summary:
            st.info(latest.summary)
        if latest.warnings:
            st.caption(" · ".join(latest.warnings))

        with st.expander("How your question was understood (editable)"):
            intent_json = st.text_area(
                "QueryIntent",
                value=json.dumps(latest.intent.model_dump(mode="json"), indent=2,
                                 default=str),
                height=260,
                key=f"intent_{len(st.session_state.history)}",
            )
            if st.button("Re-run with edited intent"):
                try:
                    edited = QueryIntent.model_validate_json(intent_json)
                    with st.spinner("Re-running…"):
                        response = pipeline.ask_intent(
                            st.session_state.session_id, edited
                        )
                    st.session_state.history.append(response)
                    st.rerun()
                except CAEError as e:
                    st.error(str(e))
                except ValueError as e:
                    st.error(f"invalid intent JSON: {e}")

        with st.expander("Generated SQL"):
            st.code(latest.sql, language="sql")

        with st.expander("Cost & timings"):
            st.json({
                "tokens_in": latest.usage.input_tokens,
                "tokens_out": latest.usage.output_tokens,
                "cost_usd": round(latest.usage.cost_usd, 5),
                "stage_timings_ms": latest.stage_timings_ms,
            })
    else:
        st.write("No questions yet — start in the chat box below.")
