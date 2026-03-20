import streamlit as st
from predictor import predict_startup # <-- Predictor file connect kiya
from advisor_ai import startup_advice # <-- Sahi function import kiya
try:
    import analytics # <-- Analytics file ko connect kiya
except ImportError:
    analytics = None

st.title("🚀 Startup Success Predictor")

tab1, tab2, tab3 = st.tabs(["🚀 Predictor", "📊 Analytics", "🤖 AI Advisor"])

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

    if st.button("Predict"):
        try:
            probability = predict_startup(funding, team_size, experience, market)
            st.session_state.prediction_prob = probability
        except Exception as e:
            st.error(f"⚠️ Error: Make sure startup_model.pkl exists (in project root or model/). Details: {e}")

    if "prediction_prob" in st.session_state:
        prob = st.session_state.prediction_prob
        st.info(f"📊 Success Probability: {prob}%")
        if prob > 50:
            st.success("✅ Looks like a solid startup!")
        else:
            st.warning("⚠️ High risk. Focus on team and product-market fit.")

with tab2:
    if analytics:
        try:
            analytics.show_dashboard()
        except Exception as e:
            st.error(f"⚠️ Analytics data load karne mein error: {e}")
    else:
        st.info("Analytics module abhi ready nahi hai.")

with tab3:
    st.header("🤖 AI Startup Advisor")
    st.caption("Ask me anything about business, funding, or validation!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Ask about your startup idea..."):
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."): # <-- Naya loading animation
                response = startup_advice(user_input)
                st.write(response)
        st.session_state.messages.append({"role":"assistant","content":response})
