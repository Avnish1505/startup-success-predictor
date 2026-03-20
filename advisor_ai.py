import streamlit as st

def fallback_response(question):
    q = question.lower()

    if "funding" in q:
        return "Focus on MVP and approach angel investors."
    elif "team" in q:
        return "Build a strong founding team."
    else:
        return "Work on product-market fit."


def startup_advice(question):

    # 1️⃣ Try Gemini
    try:
        from google import genai

        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            return "⚠️ API Key missing in Streamlit secrets."

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=question
        )

        return response.text

    except Exception as e:
        print("Gemini failed:", e)

    # 2️⃣ Try HuggingFace
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1")

        return client.text_generation(question, max_new_tokens=200)

    except Exception as e:
        print("HF failed:", e)

    # 3️⃣ Final fallback
    return fallback_response(question)