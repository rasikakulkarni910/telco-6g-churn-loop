"""
Enterprise Streamlit dashboard for the telco 6G churn control center.

Views:
  1. Executive Dashboard
  2. Segments & Drivers
  3. Loop Performance & Experiments
  4. User Explorer & Detail  ← numeric risk column (e.g. 0.78), not high/low

Requires the FastAPI backend:
  uvicorn app.api:app --port 8080
  streamlit run app/ui.py
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

DEFAULT_API = os.environ.get(
    "API_BASE_URL",
    "https://telco-6g-churn-loop-454334461204.us-central1.run.app",
)

st.set_page_config(
    page_title="6G Churn Control Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Control-center styling — flat, enterprise; avoid generic purple/AI-slop themes.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; max-width: 1400px; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
      .risk-high { color: #b42318; font-weight: 700; }
      .risk-mid { color: #b54708; font-weight: 600; }
      .risk-low { color: #027a48; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_base() -> str:
    return st.session_state.get("api_base", DEFAULT_API).rstrip("/")


def api_get(path: str, timeout: int = 60) -> dict[str, Any]:
    r = requests.get(f"{api_base()}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict[str, Any] | None = None, timeout: int = 600) -> dict[str, Any]:
    r = requests.post(f"{api_base()}{path}", json=payload or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def users_frame(users: list[dict[str, Any]]) -> pd.DataFrame:
    if not users:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "risk",
                "migration_flag",
                "value_segment",
                "top_driver_category",
                "segment",
                "playbook",
                "intensity",
                "status",
                "arm",
                "outcome_churn",
            ]
        )
    rows = []
    for u in users:
        drivers = u.get("drivers") or []
        driver_str = ", ".join(
            f"{d.get('feature')}({float(d.get('shap_value', 0)):.2f})" for d in drivers[:3]
        )
        rows.append(
            {
                "customer_id": u.get("customer_id"),
                "risk": round(float(u.get("risk") or 0.0), 4),
                "migration_flag": u.get("migration_flag"),
                "value_segment": u.get("value_segment"),
                "top_driver_category": u.get("top_driver_category"),
                "drivers_top3": driver_str,
                "segment": u.get("segment"),
                "playbook": u.get("playbook"),
                "intensity": u.get("intensity"),
                "status": u.get("status"),
                "arm": u.get("arm"),
                "outcome_churn": u.get("outcome_churn"),
                "outcome_usage_change": u.get("outcome_usage_change"),
                "region": u.get("region"),
                "device_type": u.get("device_type"),
                "intervention_channel": u.get("intervention_channel"),
                "intervention_message": u.get("intervention_message"),
            }
        )
    return pd.DataFrame(rows)


# --- Sidebar ---
with st.sidebar:
    st.title("6G Churn Loop")
    st.caption("Migration retention control center")
    st.session_state["api_base"] = st.text_input("API base URL", value=DEFAULT_API)
    limit = st.number_input("Score limit (rows)", min_value=50, max_value=5000, value=500, step=50)
    risk_threshold = st.slider("Entry risk threshold", 0.0, 1.0, 0.60, 0.05)
    seed = st.number_input("Experiment seed", min_value=0, value=42, step=1)

    col_a, col_b = st.columns(2)
    with col_a:
        run_clicked = st.button("Run full loop", type="primary", use_container_width=True)
    with col_b:
        refresh_clicked = st.button("Refresh /users", use_container_width=True)

    if run_clicked:
        with st.spinner("Running Signal → Decision → Outreach → Learning…"):
            try:
                result = api_post(
                    "/loop/run",
                    {
                        "limit": int(limit),
                        "risk_threshold": float(risk_threshold),
                        "seed": int(seed),
                    },
                )
                st.session_state["loop_result"] = result
                st.session_state["users"] = result.get("users", [])
                st.session_state["uplift"] = result.get("uplift", [])
                st.success(f"Loop complete — {result.get('n_users', 0)} eligible users")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Loop failed: {exc}")

    if refresh_clicked:
        try:
            snap = api_get("/users")
            st.session_state["users"] = snap.get("users", [])
            st.session_state["uplift"] = snap.get("uplift", [])
            st.info(f"Loaded {snap.get('n_users', 0)} users (last_step={snap.get('last_step')})")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Refresh failed: {exc}")

    st.divider()
    try:
        health = api_get("/health", timeout=5)
        st.success(f"API {health.get('status', 'ok')}")
    except Exception:
        st.warning("API offline — start: `uvicorn app.api:app --port 8080`")


users = st.session_state.get("users") or []
uplift = st.session_state.get("uplift") or []
df_users = users_frame(users)

tab_exec, tab_seg, tab_loop, tab_users = st.tabs(
    ["Executive Dashboard", "Segments & Drivers", "Loop Performance", "User Explorer"]
)

# ----- Executive -----
with tab_exec:
    st.header("Executive Dashboard")
    st.caption("C-level snapshot of migrated-cohort risk and intervention throughput")

    pop: dict[str, Any] = {}
    try:
        pop = api_get("/metrics/population", timeout=120)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Population metrics unavailable: {exc}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers", f"{int(pop.get('n_customers') or 0):,}")
    k2.metric("6G migrants", f"{int(pop.get('n_migrants') or 0):,}")
    k3.metric(
        "Churn 90d migrant",
        f"{100 * float(pop.get('migrant_churn_90d') or 0):.1f}%",
        delta=f"vs non {100 * float(pop.get('non_migrant_churn_90d') or 0):.1f}%",
    )
    k4.metric(
        "Revenue at risk (90d)",
        f"${float(pop.get('revenue_at_risk_90d') or 0):,.0f}",
    )
    k5.metric("Migration NPS proxy", f"{pop.get('migration_nps_proxy', '—')}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Loop snapshot")
        st.metric("At-risk migrants in loop", len(df_users))
        if not df_users.empty and "risk" in df_users.columns:
            st.metric("Mean numeric risk", f"{df_users['risk'].mean():.3f}")
            st.metric("Max risk", f"{df_users['risk'].max():.3f}")
    with c2:
        st.subheader("Interventions by playbook")
        if not df_users.empty and df_users["playbook"].notna().any():
            st.bar_chart(df_users["playbook"].value_counts())
        else:
            st.info("Run the loop to populate interventions.")
    with c3:
        st.subheader("Treated vs control churn")
        if uplift:
            u = pd.DataFrame(uplift)
            if u["churn_treated"].notna().any() and u["churn_control"].notna().any():
                treated = float(u["churn_treated"].dropna().mean())
                control = float(u["churn_control"].dropna().mean())
                st.metric("Control churn", f"{100 * control:.1f}%")
                st.metric(
                    "Treated churn",
                    f"{100 * treated:.1f}%",
                    delta=f"{100 * (treated - control):.1f} pp",
                    delta_color="inverse",
                )
            else:
                st.info("Need both arms in strata for uplift.")
        else:
            st.info("No experiment metrics yet.")

    st.subheader("Trend proxies (from current loop cohort)")
    if not df_users.empty:
        bands = pd.cut(
            df_users["risk"],
            bins=[0, 0.6, 0.75, 0.9, 1.01],
            labels=["<0.60", "0.60–0.75", "0.75–0.90", "0.90–1.00"],
        )
        trend = df_users.assign(risk_band=bands).groupby("risk_band", observed=False).size()
        st.bar_chart(trend)
        st.caption("Source: in-memory loop users · risk = P(churn in next 90 days)")
    else:
        st.info("Run the loop to see cohort risk distribution.")

# ----- Segments & Drivers -----
with tab_seg:
    st.header("Segments & Drivers")
    left, right = st.columns(2)
    with left:
        st.subheader("Segment breakdown")
        if not df_users.empty and df_users["segment"].notna().any():
            seg = (
                df_users.groupby("segment", dropna=False)
                .agg(
                    n=("customer_id", "count"),
                    mean_risk=("risk", "mean"),
                    churn_outcome=("outcome_churn", "mean"),
                )
                .sort_values("n", ascending=False)
            )
            seg["mean_risk"] = seg["mean_risk"].round(4)
            st.dataframe(seg, use_container_width=True)
            st.bar_chart(seg["n"].head(15))
        else:
            st.info("No segment data — run the loop.")

    with right:
        st.subheader("Driver importance (global SHAP)")
        try:
            shap = api_get("/metrics/shap")
            cats = pd.DataFrame(shap.get("categories") or [])
            if not cats.empty:
                cats = cats.set_index("driver_category")["mean_abs_shap"]
                st.bar_chart(cats)
            feats = pd.DataFrame(shap.get("features") or []).head(12)
            if not feats.empty:
                st.dataframe(
                    feats.rename(columns={"mean_abs_shap": "mean_|SHAP|"}),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"SHAP summary unavailable: {exc}")

    st.subheader("Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        seg_filter = st.multiselect(
            "Segment",
            options=sorted(df_users["segment"].dropna().unique()) if not df_users.empty else [],
        )
    with f2:
        play_filter = st.multiselect(
            "Playbook",
            options=sorted(df_users["playbook"].dropna().unique()) if not df_users.empty else [],
        )
    with f3:
        region_filter = st.multiselect(
            "Region",
            options=sorted(df_users["region"].dropna().unique()) if not df_users.empty else [],
        )
    filtered = df_users.copy()
    if seg_filter:
        filtered = filtered[filtered["segment"].isin(seg_filter)]
    if play_filter:
        filtered = filtered[filtered["playbook"].isin(play_filter)]
    if region_filter:
        filtered = filtered[filtered["region"].isin(region_filter)]
    if not filtered.empty:
        st.dataframe(
            filtered[
                [
                    "customer_id",
                    "risk",
                    "segment",
                    "playbook",
                    "top_driver_category",
                    "region",
                    "device_type",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

# ----- Loop performance -----
with tab_loop:
    st.header("Loop Performance & Experiments")
    st.caption("Treated vs control uplift by playbook and segment (simulated outcomes in v1)")
    if uplift:
        u_df = pd.DataFrame(uplift)
        for col in ["churn_control", "churn_treated", "churn_uplift", "usage_control", "usage_treated", "usage_uplift"]:
            if col in u_df.columns:
                u_df[col] = pd.to_numeric(u_df[col], errors="coerce").round(4)
        st.dataframe(u_df, use_container_width=True, hide_index=True)

        chart_df = u_df.dropna(subset=["churn_uplift"]).copy()
        if not chart_df.empty:
            chart_df["label"] = chart_df["playbook"].str.slice(0, 22) + " | " + chart_df["segment"].str.slice(0, 18)
            st.subheader("Churn reduction uplift (control − treated)")
            st.bar_chart(chart_df.set_index("label")["churn_uplift"])
            st.subheader("Usage-change uplift (treated − control)")
            st.bar_chart(chart_df.set_index("label")["usage_uplift"])
    else:
        st.info("Run the loop to compute experiment metrics.")

# ----- User explorer -----
with tab_users:
    st.header("User Explorer & Detail")
    st.caption("Numeric **risk** = P(churn in next 90 days) from XGBoost `predict_proba`")

    if df_users.empty:
        st.info("No users in API state. Click **Run full loop** in the sidebar.")
    else:
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            risk_min, risk_max = st.slider("Risk band", 0.0, 1.0, (0.0, 1.0), 0.01)
        with e2:
            status_f = st.multiselect(
                "Status",
                options=sorted(df_users["status"].dropna().unique()),
                default=list(df_users["status"].dropna().unique()),
            )
        with e3:
            driver_f = st.multiselect(
                "Driver category",
                options=sorted(df_users["top_driver_category"].dropna().unique()),
            )
        with e4:
            seg_f = st.multiselect(
                "Segment",
                options=sorted(df_users["segment"].dropna().unique()),
            )

        view = df_users[
            (df_users["risk"] >= risk_min)
            & (df_users["risk"] <= risk_max)
            & (df_users["status"].isin(status_f) if status_f else True)
        ].copy()
        if driver_f:
            view = view[view["top_driver_category"].isin(driver_f)]
        if seg_f:
            view = view[view["segment"].isin(seg_f)]

        # Explicit numeric formatting for risk — never high/low labels.
        display = view[
            [
                "customer_id",
                "risk",
                "migration_flag",
                "value_segment",
                "top_driver_category",
                "drivers_top3",
                "segment",
                "playbook",
                "intensity",
                "status",
                "arm",
                "outcome_churn",
            ]
        ].sort_values("risk", ascending=False)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "risk": st.column_config.NumberColumn(
                    "risk (P90d)",
                    help="Probability of churn in the next 90 days",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "migration_flag": st.column_config.NumberColumn("migrated", format="%d"),
            },
        )

        st.subheader("Detail panel")
        ids = display["customer_id"].tolist()
        selected = st.selectbox("Select customer", options=ids)
        detail = next((u for u in users if u.get("customer_id") == selected), None)
        if detail:
            d1, d2, d3 = st.columns(3)
            d1.metric("risk", f"{float(detail.get('risk') or 0):.4f}")
            d2.metric("playbook", str(detail.get("playbook") or "—"))
            d3.metric("status / arm", f"{detail.get('status')} / {detail.get('arm')}")

            st.markdown("**SHAP drivers**")
            drv = pd.DataFrame(detail.get("drivers") or [])
            if not drv.empty:
                st.bar_chart(drv.set_index("feature")["shap_value"])
                st.dataframe(drv, use_container_width=True, hide_index=True)

            st.markdown("**Intervention**")
            st.write(detail.get("intervention_channel"))
            st.write(detail.get("intervention_message"))
