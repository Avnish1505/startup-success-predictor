import streamlit as st
import analytics
from advisor_ai import startup_advice
from predictor import predict_startup # Local ultra-fast predictor import
import plotly.express as px

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

    col1, col2 = st.columns(2)
    with col1:
        funding = st.number_input("Funding Amount ($)", min_value=0, value=100000, step=10000)
        team_size = st.number_input("Team Size", min_value=1, value=5, step=1)
    with col2:
        experience = st.number_input("Founders Experience (Years)", min_value=0, value=2, step=1)
        market = st.selectbox("Market Size", [0, 1, 2], help="0: Small, 1: Medium, 2: Large")

    # Process prediction when button is clicked
    if st.button("Predict"):
        try:
            prob = predict_startup(funding, team_size, experience, market)
            st.session_state.prediction_prob = prob
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")

    if "prediction_prob" in st.session_state:
        prob = st.session_state.prediction_prob
        st.markdown(f"""
        <div style='padding:20px; border-radius:12px; background:#1e293b'>
        <h3>📊 Success Probability</h3>
        <h1 style='color:#22c55e;'>{prob:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(prob))

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🧠 Why this result?")
            reasons = []
            if funding > 150000: reasons.append("💰 Strong funding boosts success chances")
            if experience > 2: reasons.append("👨‍💼 Experienced founders improve execution")
            if team_size > 5: reasons.append("👥 Larger team supports scaling")
            if market == 2: reasons.append("🌍 Large market increases opportunity")
            
            for r in reasons: st.write("- " + r)
                
            st.subheader("🎯 Suggestions")
            if prob < 50:
                st.write("• Increase funding\n• Improve team experience\n• Focus on product-market fit")
            else:
                st.write("• Scale your startup\n• Expand into new markets")

        with col_b:
            fig = px.pie(values=[prob, 100-prob], names=["Success", "Failure"], title="Prediction Split", color_discrete_sequence=['#22c55e', '#ef4444'])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Industry Comparison")
        st.info(f"Your startup performs better than {int(prob)}% of similar startups")
        
        report_data = f"Startup Success Probability: {prob:.2f}%\nFunding: ${funding}\nTeam: {team_size}\nExperience: {experience} yrs\nMarket: {market}"
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
