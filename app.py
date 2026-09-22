import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import analytics
from advisor_ai import startup_advice
import plotly.express as px

from src.data import schema
from src.models.explain import explain_prediction
from src.models.percentile import compute_percentile, load_reference_distribution

PRODUCTION_DIR = Path("models/production")
FEATURE_COLS = schema.FULL_NUMERIC_FEATURES + schema.FULL_CATEGORICAL_FEATURES


@st.cache_resource
def load_production_artifacts():
    calibrated = joblib.load(PRODUCTION_DIR / "hist_gradient_boosting_full_calibrated.joblib")
    base = joblib.load(PRODUCTION_DIR / "hist_gradient_boosting_full_calibrated_base.joblib")
    reference = load_reference_distribution(PRODUCTION_DIR / "train_score_distribution.npy")
    options = json.loads((PRODUCTION_DIR / "category_options.json").read_text())
    return calibrated, base, reference, options


st.set_page_config(
    page_title="Startup AI",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #333333; /* Premium dark grey color */
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e5e5;
}

/* Chat container */
.chat-container {
    max-width: 700px;
    margin: auto;
}

/* User message */
.chat-user {
    background: #10a37f;
    color: white;
    padding: 12px 15px;
    border-radius: 12px 12px 0px 12px;
    margin: 8px 0;
    width: fit-content;
    margin-left: auto;
    font-size: 14px;
}

/* Bot message */
.chat-bot {
    background: #1e293b;
    border: 1px solid #334155;
    color: white;
    padding: 12px 15px;
    border-radius: 12px 12px 12px 0px;
    margin: 8px 0;
    width: fit-content;
    font-size: 14px;
}

/* Input box */
.stTextInput>div>div>input {
    background-color: white;
    color: black; /* Ensures text is visible (black) inside the white input box */
    border-radius: 12px;
    padding: 10px;
}

/* Button */
.stButton button {
    background-color: #10a37f;
    color: white;
    border-radius: 10px;
    padding: 8px 16px;
}

.chat-bot {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Predictor", "📊 Analytics", "🤖 AI Advisor"])

# ---------------- Predictor ----------------
with tab1:
    st.header("Startup Predictor")
    st.write("Enter startup details to predict success probability")

    calibrated_pipeline, base_pipeline, train_reference, category_options = load_production_artifacts()

    col1, col2 = st.columns(2)
    with col1:
        founded_date = st.date_input("Founded date", value=date(2013, 1, 1))
        first_funding_date = st.date_input("First funding date", value=date(2013, 6, 1))
        last_funding_date = st.date_input("Most recent funding date", value=date(2014, 1, 1))
        funding_total_usd = st.number_input("Total funding raised ($)", min_value=0, value=1_000_000, step=10_000)
        funding_rounds = st.number_input("Number of funding rounds", min_value=1, value=2, step=1)
    with col2:
        country_options = category_options["country_code"] + ["UNKNOWN"]
        country_code = st.selectbox(
            "Country", country_options,
            index=country_options.index("USA") if "USA" in country_options else 0,
        )
        region = st.selectbox("Region", category_options["region"] + ["UNKNOWN"])
        primary_category = st.selectbox("Primary category", category_options["primary_category"] + ["UNKNOWN"])

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

    # Process prediction when button is clicked
    if st.button("Predict"):
        try:
            prob = float(calibrated_pipeline.predict_proba(input_row[FEATURE_COLS])[0, 1]) * 100
            st.session_state.prediction_prob = prob
            st.session_state.prediction_input = input_row
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")

    if "prediction_prob" in st.session_state:
        prob = st.session_state.prediction_prob
        st.markdown(f"""
        <div style='padding:20px; border-radius:12px; background:#1e293b'>
        <h3>📊 Success Probability</h3>
        <h1 style='color:#22c55e;'>{prob:.2f}%</h1>
        <p style='color:#94a3b8; font-size:12px;'>Calibrated via CalibratedClassifierCV - see MODEL_CARD.md</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(prob))

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🧠 Why this result?")
            contributions = explain_prediction(
                base_pipeline, FEATURE_COLS, st.session_state.prediction_input, top_n=5
            )
            for c in contributions:
                arrow = "⬆️" if c["direction"] == "increases" else "⬇️"
                st.write(f"- {arrow} `{c['feature']}` = {c['value']} {c['direction']} predicted success (SHAP {c['shap_value']:+.3f})")

            st.subheader("🎯 Suggestions")
            if prob < 50:
                st.write("• Shorten time-to-first-funding\n• Pursue additional funding rounds\n• Target regions/categories with stronger historical outcomes")
            else:
                st.write("• Maintain funding momentum\n• Expand into additional high-performing regions")

        with col_b:
            fig = px.pie(values=[prob, 100-prob], names=["Success", "Failure"], title="Prediction Split", color_discrete_sequence=['#22c55e', '#ef4444'])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Industry Comparison")
        percentile = compute_percentile(prob / 100.0, train_reference)
        st.info(f"Your predicted score is higher than {percentile:.1f}% of startups in the training cohort (calibrated empirical percentile, not the raw probability)")

        report_data = (
            f"Startup Success Probability (calibrated): {prob:.2f}%\n"
            f"Percentile vs. training cohort: {percentile:.1f}%\n"
            f"Founded: {founded_date} | First funding: {first_funding_date} | Last funding: {last_funding_date}\n"
            f"Total funding: ${funding_total_usd} | Rounds: {funding_rounds}\n"
            f"Country: {country_code} | Region: {region} | Category: {primary_category}"
        )
        st.download_button(label="📄 Download Report", data=report_data, file_name="startup_report.txt")

# ---------------- Analytics ----------------
with tab2:
    analytics.show_dashboard()

# ---------------- AI Advisor ----------------
with tab3:
    st.header("🤖 AI Startup Advisor")
    st.caption("💡 Ask anything about your startup idea, funding, or growth strategy")

    # Initialize custom chat session state variables
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "pending_msg" not in st.session_state:
        st.session_state.pending_msg = ""

    def submit_message():
        if st.session_state.chat_box:
            st.session_state.pending_msg = st.session_state.chat_box
            st.session_state.chat_box = "" # Clear the input box after submission

    # 1. Process pending messages without full page refresh
    if st.session_state.pending_msg:
        msg = st.session_state.pending_msg
        st.session_state.pending_msg = "" # Immediate reset to prevent loops
        with st.spinner("AI is thinking..."):
            response = startup_advice(msg)
            st.session_state.chat.append((msg, response))

    # 2. Render complete chat history including new messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for user, bot in st.session_state.chat:
        st.markdown(f'<div class="chat-user">{user}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bot">{bot}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Display the input box at the bottom
    col1, col2 = st.columns([4, 1])
    with col1:
        st.text_input("Ask...", key="chat_box", label_visibility="collapsed", placeholder="Ask about your startup idea...", on_change=submit_message)
    with col2:
        st.button("Send", on_click=submit_message, use_container_width=True)
