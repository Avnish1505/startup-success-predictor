import streamlit as st
import pandas as pd
import plotly.express as px


def show_dashboard():
    """Renders the analytics dashboard with metrics and Plotly visualizations."""
    st.title("📊 Startup Analytics Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Success Rate", "72%")
    col2.metric("Top Market", "E-commerce")
    col3.metric("Risk Level", "Medium")

    data = pd.DataFrame({
        "Market": ["Tech", "Healthcare", "Finance", "E-commerce"],
        "Success Rate": [75, 65, 70, 80]
    })

    fig = px.bar(data, x="Market", y="Success Rate", color="Market",
                 title="Market-wise Success Rate")

    st.plotly_chart(fig)

    df2 = pd.DataFrame({
        "Funding": [50, 100, 200, 300, 500],
        "Success": [40, 55, 65, 75, 85]
    })

    fig2 = px.line(df2, x="Funding", y="Success",
                   title="Funding vs Success")

    st.plotly_chart(fig2)
    
    fig3 = px.pie(data, names="Market", values="Success Rate", title="Market Success Distribution")
    st.plotly_chart(fig3)