"""Streamlit operations dashboard for the Beaver's Choice multi-agent system.

A portfolio / demo layer on top of `template.py` — it reuses the exact same
agents, tools, and database, and is NOT part of the single-file course submission.

Run from this directory (with `.env` configured), in the project venv:

    .venv/bin/python -m streamlit run app.py

Requires the project deps plus `streamlit` (`pip install streamlit`).
"""
import os

import pandas as pd
import streamlit as st

import template as t

# A date safely after every sample request (April-May 2025) so the financial
# snapshot reflects all activity recorded in the database so far.
AS_OF_DATE = "2025-12-31"
DEFAULT_REQUEST_DATE = "2025-04-01"
LOW_STOCK_THRESHOLD = 100

st.set_page_config(page_title="Beaver's Choice — Operations", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Initializing database and agents…")
def get_orchestrator():
    """Seed the database and build the 5-agent system once per session."""
    t.init_database(t.db_engine)
    return t.Orchestrator(t.model)


@st.cache_data
def load_sample_requests():
    df = pd.read_csv("quote_requests_sample.csv")
    df["request_date"] = pd.to_datetime(df["request_date"], format="%m/%d/%y", errors="coerce")
    return df.dropna(subset=["request_date"]).sort_values("request_date").reset_index(drop=True)


def render_sidebar():
    """Live financial dashboard — the Business Advisor agent's data, on screen."""
    st.sidebar.header("📊 Business dashboard")
    report = t.generate_financial_report(AS_OF_DATE)
    st.sidebar.metric("Cash balance", f"${report['cash_balance']:,.2f}")
    st.sidebar.metric("Inventory value", f"${report['inventory_value']:,.2f}")
    st.sidebar.metric("Total assets", f"${report['total_assets']:,.2f}")

    top = pd.DataFrame(report.get("top_selling_products", []))
    if not top.empty:
        # Drop the initial cash-seed row: starting cash is recorded as a sale with a
        # null item_name, which otherwise sorts to the top by revenue.
        top = top.dropna(subset=["item_name"])
    if not top.empty:
        top["total_revenue"] = top["total_revenue"].round(2)
        st.sidebar.subheader("Top sellers")
        st.sidebar.dataframe(top, hide_index=True, width="stretch")

    inventory = t.get_all_inventory(AS_OF_DATE)
    low = sorted(((v, k) for k, v in inventory.items() if v < LOW_STOCK_THRESHOLD))
    if low:
        st.sidebar.subheader("⚠️ Low stock")
        st.sidebar.dataframe(pd.DataFrame([{"item": k, "units": v} for v, k in low]),
                             hide_index=True, width="stretch")

    st.sidebar.divider()
    if st.sidebar.button("🔄 Reset simulation", help="Reseed the database to its initial state"):
        t.init_database(t.db_engine)
        st.session_state.pop("history", None)
        st.cache_resource.clear()
        st.rerun()


def render_live_tab(orchestrator, samples):
    """Submit a customer request and watch the agents handle it."""
    st.subheader("Submit a customer request")
    col1, col2 = st.columns([3, 1])
    with col2:
        request_date = st.text_input("Request date (YYYY-MM-DD)", DEFAULT_REQUEST_DATE)
        preset = st.selectbox(
            "Load a sample request",
            ["(write my own)"] + [f"#{i+1}: {r.job} / {r.event}" for i, r in enumerate(samples.itertuples())],
        )
    default_text = ""
    if preset != "(write my own)":
        default_text = samples.iloc[int(preset.split(":")[0][1:]) - 1]["request"]
    with col1:
        request_text = st.text_area("Request", value=default_text, height=160,
                                    placeholder="e.g. I'd like 500 sheets of A4 glossy paper for a wedding…")

    if st.button("Process request", type="primary", disabled=not request_text.strip()):
        prompt = f"{request_text.strip()} (Date of request: {request_date})"
        with st.spinner("Agents are working: inventory → quoting → sales…"):
            reply = orchestrator.process_request(prompt)
        summary = orchestrator.last_summary
        st.session_state.setdefault("history", []).insert(0, (request_text.strip(), reply, summary))

    for req, reply, summary in st.session_state.get("history", []):
        fulfilled = summary.get("fulfilled")
        with st.container(border=True):
            st.markdown(f"**Customer:** {req}")
            (st.success if fulfilled else st.warning)(reply)
            with st.expander("Behind the scenes (internal)"):
                cols = st.columns(4)
                cols[0].metric("Lines requested", summary.get("num_lines_requested", 0))
                cols[1].metric("Fulfilled", summary.get("num_lines_fulfilled", 0))
                cols[2].metric("Declined", summary.get("num_lines_declined", 0))
                cols[3].metric("Charged", f"${summary.get('total_charged', 0):,.2f}")
                if summary.get("decline_reasons"):
                    st.caption(f"Decline reasons: {summary['decline_reasons']}")


def render_results_tab():
    """Visualize the committed evaluation output (test_results.csv)."""
    if not os.path.exists("test_results.csv"):
        st.info("No `test_results.csv` yet — run `python template.py` to generate it.")
        return
    r = pd.read_csv("test_results.csv")

    c = st.columns(4)
    c[0].metric("Requests", len(r))
    c[1].metric("Fulfilled", int((r["fulfilled"] == 1).sum()))
    c[2].metric("With declines", int((r["num_lines_declined"] > 0).sum()))
    c[3].metric("Total revenue", f"${r['total_charged'].sum():,.2f}")

    st.subheader("Cash & inventory over the run")
    st.line_chart(r.set_index("request_id")[["cash_balance", "inventory_value"]])

    st.subheader("Fulfilment outcome by request")
    outcome = pd.Series({
        "Fully fulfilled": int(((r["fulfilled"] == 1) & (r["num_lines_declined"] == 0)).sum()),
        "Partial": int(((r["num_lines_fulfilled"] > 0) & (r["num_lines_declined"] > 0)).sum()),
        "Declined": int(((r["num_lines_fulfilled"] == 0) & (r["num_lines_declined"] > 0)).sum()),
    })
    st.bar_chart(outcome)

    st.subheader("All results")
    st.dataframe(r, hide_index=True, width="stretch")


def main():
    st.title("📄 Beaver's Choice Paper Company — Operations")
    st.caption("Multi-agent order handling (smolagents + gpt-4o-mini): inventory, "
               "retrieval-augmented quoting, and sales finalization.")
    orchestrator = get_orchestrator()
    samples = load_sample_requests()
    render_sidebar()
    live, results = st.tabs(["🛒 Live request", "📈 Evaluation results"])
    with live:
        render_live_tab(orchestrator, samples)
    with results:
        render_results_tab()


if __name__ == "__main__":
    main()
