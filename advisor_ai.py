import streamlit as st

def fallback_response(question):
    """
    Provides a rule-based fallback response if the AI API fails.
    
    Args:
        question (str): The user's query.
    """
    q = question.lower()

    if "funding" in q:
        return "Focus on MVP and approach angel investors."
    elif "team" in q:
        return "Build a strong founding team."
    else:
        return "Work on product-market fit."


def startup_advice(question):
    """
    Fetches startup advice from the Google Gemini AI model.
    Falls back to a basic response mechanism upon failure.
    
    Args:
        question (str): The user's startup-related query.
    """
    try:
        from google import genai

        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            return "⚠️ API Key missing in Streamlit secrets."

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=question
        )
        return response.text

    except Exception as e:
        # Log error to UI and provide a fallback response upon API failure
        return f"⚠️ **Gemini AI Error:** {e}\n\n💡 *Basic Advice:* {fallback_response(question)}"