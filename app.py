import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics
import theme
from advisor_ai import startup_advice
from src.models.confidence import bootstrap_confidence_band
from src.models.explain import explain_prediction
from src.models.partial_dependence import compute_partial_dependence
from src.models.percentile import compute_percentile, load_reference_distribution

PRODUCTION_DIR = Path("models/production")
DATA_PATH = Path("data/processed/startups_features_v1.parquet")
MODEL_NAMES = {"clean": "hist_gradient_boosting_clean_calibrated", "full": "hist_gradient_boosting_full_calibrated"}


@st.cache_resource
def load_production_artifacts():
    models = {}
    for feature_set, name in MODEL_NAMES.items():
        metadata = json.loads((PRODUCTION_DIR / f"{name}_metadata.json").read_text())
        calib_data = np.load(PRODUCTION_DIR / f"{name}_calibration_raw_scores.npz")
        models[feature_set] = {
            "calibrated": joblib.load(PRODUCTION_DIR / f"{name}.joblib"),
            "base": joblib.load(PRODUCTION_DIR / f"{name}_base.joblib"),
            "reference": load_reference_distribution(PRODUCTION_DIR / f"{name}_train_score_distribution.npy"),
            "metadata": metadata,
            "feature_cols": metadata["numeric_features"] + metadata["categorical_features"],
            "calibration_raw_scores": calib_data["raw_score"],
            "calibration_labels": calib_data["label"],
        }
    options = json.loads((PRODUCTION_DIR / "category_options.json").read_text())
    population_stats = json.loads((PRODUCTION_DIR / "population_stats.json").read_text())
    region_by_country = json.loads((PRODUCTION_DIR / "region_by_country.json").read_text())
    return models, options, population_stats, region_by_country


@st.cache_data
def load_dataset_n() -> int:
    return len(pd.read_parquet(DATA_PATH))


st.set_page_config(
    page_title="Startup Predictor",
    page_icon=":material/analytics:",
    layout="wide",
)

st.markdown(theme.inject_css(), unsafe_allow_html=True)
theme.register_plotly_template()

models, category_options, population_stats, region_by_country = load_production_artifacts()
clean_meta, full_meta = models["clean"]["metadata"], models["full"]["metadata"]
dataset_n = load_dataset_n()

tab1, tab2, tab3 = st.tabs(["Predictor", "Analytics", "Advisor"])

# ---------------- Predictor ----------------
with tab1:
    st.header("Startup outcome predictor")
    st.caption("Enter a funding and geography profile to get a calibrated probability, not a bare guess.")

    feature_set_choice = st.radio(
        "Feature set", ["clean", "full (leakage demonstration)"], horizontal=True,
    )
    feature_set = "full" if feature_set_choice.startswith("full") else "clean"
    active = models[feature_set]
    model_metadata = active["metadata"]

    st.markdown(
        f"""
        <div class="status-strip">
            <span>MODEL <b>{model_metadata['model_name']}</b></span>
            <span>N <b>{dataset_n:,}</b></span>
            <span>TEST AUC <b>{model_metadata['roc_auc_after']:.3f}</b></span>
            <span>CALIBRATION <b>{model_metadata['calibration_method']}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="readout-sub">Clean AUC <b>{clean_meta["roc_auc_after"]:.3f}</b> &middot; '
        f'Full AUC <b>{full_meta["roc_auc_after"]:.3f}</b> &mdash; full includes funding fields measured '
        f'after the outcome resolved (post-acquisition funding, final round count), which inflates its '
        f'AUC. See MODEL_CARD.md. Clean is the default because it is the only one honest for scoring an '
        f'active, unresolved company.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        founded_date = st.date_input("Founded date", value=date(2013, 1, 1))
        first_funding_date = st.date_input("First funding date", value=date(2013, 6, 1))
        last_funding_date = st.date_input("Most recent funding date", value=date(2014, 1, 1))
        funding_total_usd = st.number_input("Total funding raised (USD)", min_value=0, value=1_000_000, step=10_000)
        funding_rounds = st.number_input("Number of funding rounds", min_value=1, value=2, step=1)
        if feature_set == "clean":
            st.caption("Funding fields above are collected but not used by the clean model - switch to the full (leakage demonstration) toggle to see their effect.")
    with col2:
        country_options = category_options["country_code"] + ["UNKNOWN"]
        country_code = st.selectbox(
            "Country", country_options,
            index=country_options.index("USA") if "USA" in country_options else 0,
        )
        country_regions = region_by_country.get(country_code, [])
        region_options = country_regions + ["UNKNOWN"] if country_regions else category_options["region"] + ["UNKNOWN"]
        region = st.selectbox("Region", region_options)  # sorted by frequency for this country - most common first
        primary_category = st.selectbox("Primary category", category_options["primary_category"] + ["UNKNOWN"])
        confidence_threshold = st.slider(
            "Confidence threshold for a stated decision", min_value=50, max_value=99, value=65, step=1,
            help="Below this, the app abstains rather than forcing a call.",
        )

    input_row = pd.DataFrame([{
        "founded_year": float(founded_date.year),
        "time_to_first_funding_days": float((first_funding_date - founded_date).days),
        "funding_total_usd_log1p": float(np.log1p(funding_total_usd)),
        "funding_rounds": float(funding_rounds),
        "funding_span_days": float((last_funding_date - first_funding_date).days),
        "country_code": country_code,
        "region": region,
        "primary_category": primary_category,
    }])

    if st.button("Predict"):
        try:
            feature_cols = active["feature_cols"]
            prob = float(active["calibrated"].predict_proba(input_row[feature_cols])[0, 1]) * 100
            st.session_state.prediction_prob = prob
            st.session_state.prediction_input = input_row
            st.session_state.prediction_feature_set = feature_set
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if "prediction_prob" in st.session_state and st.session_state.get("prediction_feature_set") == feature_set:
        prob = st.session_state.prediction_prob
        feature_cols = active["feature_cols"]

        raw_prob = float(active["base"].predict_proba(st.session_state.prediction_input[feature_cols])[0, 1])
        try:
            ci_lower, ci_upper = bootstrap_confidence_band(
                raw_prob, active["calibration_raw_scores"], active["calibration_labels"],
                model_metadata["calibration_method"], point_estimate=prob / 100.0, n_bootstrap=200,
            )
        except AssertionError as e:
            st.error(f"Confidence interval failed an internal consistency check: {e}")
            ci_lower, ci_upper = prob / 100.0, prob / 100.0

        if prob >= confidence_threshold:
            decision_html = '<div class="decision-stated">DECISION &nbsp; likely success</div>'
        elif prob <= (100 - confidence_threshold):
            decision_html = '<div class="decision-stated">DECISION &nbsp; likely failure</div>'
        else:
            decision_html = '<div class="decision-abstain">INSUFFICIENT CONFIDENCE &nbsp; abstaining</div>'

        st.markdown(
            f"""
            <div class="readout-panel">
                <div class="readout-label">Calibrated probability</div>
                <div class="headline-number">{prob:.1f}%</div>
                <div class="readout-sub">90% CI &nbsp; {ci_lower:.1%} &ndash; {ci_upper:.1%}
                &nbsp;&nbsp;|&nbsp;&nbsp; {model_metadata['calibration_method']} calibration,
                n={model_metadata['n_calibration']:,}</div>
                {decision_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"Raw model score (pre-calibration): {raw_prob:.1%} - the bars below decompose this number, not the calibrated {prob:.1f}% above.")
            st.subheader("Signed contributions")
            contributions = explain_prediction(
                active["base"], feature_cols, st.session_state.prediction_input, top_n=5
            )
            st.session_state.prediction_shap_contributions = contributions

            shap_df = pd.DataFrame(contributions).sort_values("shap_value")
            colors = [theme.ACCENT if v > 0 else theme.FG for v in shap_df["shap_value"]]
            shap_fig = go.Figure(go.Bar(
                x=shap_df["shap_value"], y=shap_df["feature"], orientation="h",
                marker_color=colors, text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
                textposition="inside", insidetextanchor="end", textfont_color=theme.BG,
            ))
            shap_fig.update_traces(textfont_size=10)
            shap_fig.add_vline(x=0, line_color=theme.BORDER, line_width=1)
            shap_fig.update_layout(
                height=260, showlegend=False,
                yaxis={"title": "", "automargin": True},
                xaxis={"title": "SHAP value"},
            )
            theme.apply_theme(shap_fig)
            shap_fig.update_layout(margin={"t": 20, "l": 8, "r": 20, "b": 30})
            st.plotly_chart(shap_fig, use_container_width=True)

            st.subheader("Sensitivity")
            pdp_feature = st.selectbox("Vary this input, holding everything else fixed", feature_cols, key="pdp_feature")
            input_row_snapshot = st.session_state.prediction_input

            if pdp_feature in population_stats["numeric_ranges"]:
                bounds = population_stats["numeric_ranges"][pdp_feature]
                sweep_min = bounds["p01"]
                if pdp_feature == "founded_year":
                    # p01 for founded_year is really 1972.0 (measured, DATA_CARD.md) -
                    # 1972-1979 are real but sparse (1-2 rows each); floor at 1980 on
                    # top of the general p01/p99 clip for this specific feature.
                    sweep_min = max(sweep_min, 1980)
                grid = np.linspace(sweep_min, bounds["p99"], 30).tolist()
            else:
                grid = population_stats["categorical_top"][pdp_feature]

            pdp_df = compute_partial_dependence(active["calibrated"], feature_cols, input_row_snapshot, pdp_feature, grid)
            current_value = input_row_snapshot.iloc[0][pdp_feature]

            if pdp_feature in population_stats["numeric_ranges"]:
                pdp_fig = px.line(pdp_df, x=pdp_feature, y="predicted_probability", markers=True,
                                   title=f"probability vs. {pdp_feature} (1st-99th pct.)")
                pdp_fig.update_traces(line_color=theme.ACCENT, marker_color=theme.ACCENT)
                pdp_fig.add_vline(x=current_value, line_dash="dash", line_color=theme.FG, annotation_text="current")
            else:
                pdp_fig = px.bar(pdp_df, x=pdp_feature, y="predicted_probability",
                                  title=f"probability vs. {pdp_feature} (top {len(grid)} by frequency)")
                pdp_fig.update_traces(marker_color=theme.ACCENT)
            pdp_fig.update_yaxes(tickformat=".0%", range=[0, 1])
            theme.apply_theme(pdp_fig)
            st.plotly_chart(pdp_fig, use_container_width=True)

        with col_b:
            base_rate = population_stats["train_base_rate"] * 100
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%", "font": {"family": theme.FONT_MONO, "color": theme.FG}},
                title={"text": "vs. training-cohort base rate", "font": {"family": theme.FONT_MONO, "size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"family": theme.FONT_MONO}},
                    "bar": {"color": theme.ACCENT},
                    "bgcolor": theme.BG,
                    "bordercolor": theme.BORDER,
                    "threshold": {"line": {"color": theme.FG, "width": 2}, "thickness": 0.9, "value": base_rate},
                },
            ))
            theme.apply_theme(gauge_fig)
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.caption(f"Line marks the training-cohort base rate ({base_rate:.1f}%), not 50%.")

            st.subheader("Cohort standing")
            percentile = compute_percentile(prob / 100.0, active["reference"])
            st.session_state.prediction_percentile = percentile
            st.markdown(
                f'<div class="readout-sub">Higher than <b>{percentile:.1f}%</b> of startups '
                f'in the training cohort (calibrated percentile, not raw probability).</div>',
                unsafe_allow_html=True,
            )

        snap = st.session_state.prediction_input.iloc[0]
        report_data = (
            f"Startup Success Probability (calibrated, {feature_set} model): {prob:.2f}%\n"
            f"90% confidence band: {ci_lower:.1%} - {ci_upper:.1%}\n"
            f"Percentile vs. training cohort: {percentile:.1f}%\n"
            f"Founded year: {snap['founded_year']:.0f} | Time to first funding: {snap['time_to_first_funding_days']:.0f} days\n"
            f"Total funding: ${np.expm1(snap['funding_total_usd_log1p']):,.0f} | Rounds: {snap['funding_rounds']:.0f} | Funding span: {snap['funding_span_days']:.0f} days\n"
            f"Country: {snap['country_code']} | Region: {snap['region']} | Category: {snap['primary_category']}"
        )
        st.download_button(label="Download report", data=report_data, file_name="startup_report.txt")

# ---------------- Analytics ----------------
with tab2:
    analytics.show_dashboard()

# ---------------- AI Advisor ----------------
with tab3:
    st.header("Advisor")
    st.caption("Deterministic facts from your last prediction, plus sourced passages retrieved locally. No network call.")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "pending_msg" not in st.session_state:
        st.session_state.pending_msg = ""

    def submit_message():
        if st.session_state.chat_box:
            st.session_state.pending_msg = st.session_state.chat_box
            st.session_state.chat_box = ""

    if st.session_state.pending_msg:
        msg = st.session_state.pending_msg
        st.session_state.pending_msg = ""
        with st.spinner("Retrieving..."):
            response = startup_advice(msg)
            st.session_state.chat.append((msg, response))

    for user, bot in st.session_state.chat:
        with st.container(border=True):
            st.markdown('<div class="qa-label">Question</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="qa-question">{user}</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="qa-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(bot)

    col1, col2 = st.columns([4, 1])
    with col1:
        st.text_input("Ask", key="chat_box", label_visibility="collapsed", placeholder="Ask about your startup idea...", on_change=submit_message)
    with col2:
        st.button("Ask", on_click=submit_message, use_container_width=True)
