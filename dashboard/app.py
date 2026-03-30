import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from monitoring.drift_detector import run_drift_check
from monitoring.performance_tracker import compute_metrics
from monitoring.alerts import threshold_alerts, drift_alert

st.set_page_config(page_title="ML Monitoring Dashboard", page_icon="📊", layout="wide")
st.title("ML Monitoring Dashboard")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    ref_path = st.text_input(
        "Reference data CSV",
        os.path.join(BASE_DIR, "data", "reference_data.csv"),
    )
    prod_path = st.text_input(
        "Production data CSV",
        os.path.join(BASE_DIR, "data", "production_data.csv"),
    )
    model_path = st.text_input(
        "Model (.pkl)",
        os.path.join(BASE_DIR, "model", "baseline_model.pkl"),
    )

    st.subheader("Thresholds")
    psi_threshold = st.slider("PSI threshold", 0.0, 0.5, 0.1, 0.01)
    ks_threshold = st.slider("KS p-value cutoff", 0.01, 0.2, 0.05, 0.01)
    acc_threshold = st.slider("Min accuracy", 0.0, 1.0, 0.80, 0.01)


# ── Data helpers ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(ref: str, prod: str):
    return pd.read_csv(ref), pd.read_csv(prod)


@st.cache_data
def get_drift(ref: str, prod: str, psi_t: float, ks_t: float):
    return run_drift_check(ref, prod, exclude_columns=["target"], psi_threshold=psi_t, ks_p_threshold=ks_t)


try:
    ref_df, prod_df = load_data(ref_path, prod_path)
    data_ok = True
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    data_ok = False

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_drift, tab_perf, tab_alerts = st.tabs(
    ["Overview", "Data Drift", "Performance", "Alerts"]
)

if data_ok:
    drift_report = get_drift(ref_path, prod_path, psi_threshold, ks_threshold)
    drifted = [f for f, m in drift_report.items() if m.get("drift_detected")]

    # ── Overview ────────────────────────────────────────────────────────────────
    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Features monitored", len(drift_report))
        c2.metric(
            "Drifted features",
            len(drifted),
            delta=f"{len(drifted)} alert(s)" if drifted else None,
            delta_color="inverse",
        )
        psi_vals = [m["psi"] for m in drift_report.values() if "psi" in m]
        avg_psi = float(np.mean(psi_vals)) if psi_vals else 0.0
        c3.metric(
            "Avg PSI",
            f"{avg_psi:.4f}",
            delta="HIGH" if avg_psi > psi_threshold else "OK",
            delta_color="inverse" if avg_psi > psi_threshold else "off",
        )

        st.subheader("Feature distribution: reference vs production")
        num_cols = ref_df.select_dtypes(include=np.number).columns.tolist()
        sel = st.selectbox("Feature", num_cols)
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=ref_df[sel], name="Reference", opacity=0.65, nbinsx=30,
                marker_color="steelblue",
            )
        )
        fig.add_trace(
            go.Histogram(
                x=prod_df[sel], name="Production", opacity=0.65, nbinsx=30,
                marker_color="tomato",
            )
        )
        fig.update_layout(
            barmode="overlay",
            xaxis_title=sel,
            yaxis_title="Count",
            legend=dict(x=0.78, y=0.95),
        )
        st.plotly_chart(fig, width="stretch")

    # ── Data Drift ──────────────────────────────────────────────────────────────
    with tab_drift:
        st.subheader("Drift metrics by feature")
        rows = []
        for feat, m in drift_report.items():
            row = {"Feature": feat, "Drift": "YES" if m.get("drift_detected") else "NO"}
            if "psi" in m:
                row.update(
                    {
                        "PSI": round(m["psi"], 4),
                        "KS stat": round(m.get("ks_stat", 0), 4),
                        "KS p-value": round(m.get("ks_p", 1), 4),
                    }
                )
            elif "chi2" in m:
                row.update(
                    {
                        "Chi2": round(m["chi2"], 4),
                        "p-value": round(m["p_value"], 4),
                    }
                )
            rows.append(row)

        drift_df = pd.DataFrame(rows)

        def highlight_drift(row):
            color = "#ffcccc" if row["Drift"] == "YES" else ""
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            drift_df.style.apply(highlight_drift, axis=1), width="stretch"
        )

        if "PSI" in drift_df.columns:
            fig_psi = px.bar(
                drift_df,
                x="Feature",
                y="PSI",
                color="Drift",
                color_discrete_map={"YES": "tomato", "NO": "steelblue"},
                title="Population Stability Index (PSI) per feature",
            )
            fig_psi.add_hline(
                y=psi_threshold,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Threshold ({psi_threshold})",
            )
            st.plotly_chart(fig_psi, width="stretch")

            fig_ks = px.bar(
                drift_df,
                x="Feature",
                y="KS stat",
                color="Drift",
                color_discrete_map={"YES": "tomato", "NO": "steelblue"},
                title="KS statistic per feature",
            )
            st.plotly_chart(fig_ks, width="stretch")

    # ── Performance ─────────────────────────────────────────────────────────────
    with tab_perf:
        st.subheader("Model performance on production data")
        model_ok = False
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                model_ok = True
                st.success("Model loaded successfully")
            except Exception as exc:
                st.error(f"Model load failed: {exc}")
        else:
            st.warning("Model not found — run `python model/train.py` first.")

        if model_ok and "target" in prod_df.columns:
            X = prod_df.drop(columns=["target"])
            y = prod_df["target"]
            try:
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)
                metrics = compute_metrics(y, y_pred, y_proba)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric(
                    "Accuracy",
                    f"{metrics['accuracy']:.4f}",
                    delta="LOW" if metrics["accuracy"] < acc_threshold else "OK",
                    delta_color="inverse" if metrics["accuracy"] < acc_threshold else "off",
                )
                mc2.metric("F1 (weighted)", f"{metrics['f1']:.4f}")
                if metrics.get("auc"):
                    mc3.metric("AUC (OvR)", f"{metrics['auc']:.4f}")

                perf_alerts = threshold_alerts(
                    metrics, {"accuracy": acc_threshold, "f1": 0.70}
                )
                for a in perf_alerts:
                    st.warning(f"Performance alert: {a}")

            except Exception as exc:
                st.error(f"Evaluation error: {exc}")

        elif model_ok:
            st.info(
                "No 'target' column in production data — "
                "upload ground truth labels to evaluate performance."
            )

        # Evidently report embed
        report_path = os.path.join(BASE_DIR, "reports", "monitoring_report.html")
        if os.path.exists(report_path):
            st.divider()
            st.subheader("Evidently report")
            with open(report_path, encoding="utf-8") as fh:
                html = fh.read()
            st.download_button(
                "Download report", html, "monitoring_report.html", "text/html"
            )
            st.components.v1.html(html, height=600, scrolling=True)
        else:
            st.info(
                "No Evidently report found. "
                "Run `python reports/generate_report.py` to generate one."
            )

    # ── Alerts ──────────────────────────────────────────────────────────────────
    with tab_alerts:
        st.subheader("Active drift alerts")
        alerts = drift_alert(
            drift_report, psi_threshold=psi_threshold, ks_p_threshold=ks_threshold
        )
        if alerts:
            for a in alerts:
                st.error(f"ALERT: {a}")
        else:
            st.success("All features within drift thresholds — no alerts.")

else:
    with tab_overview:
        st.info("Configure data paths in the sidebar to begin monitoring.")
