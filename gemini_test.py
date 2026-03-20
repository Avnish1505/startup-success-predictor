import os

from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("Set GEMINI_API_KEY in your environment before running this script.")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Explain how AI works in a few words"
)
print(response.text)