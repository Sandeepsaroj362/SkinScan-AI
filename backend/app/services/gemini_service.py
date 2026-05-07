import os

from groq import Groq

from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_explanation(
    disease,
    confidence
):

    prompt = f"""
    You are an AI dermatologist assistant.

    Predicted disease:
    {disease}

    Confidence:
    {confidence}%

    Explain:
    - what this condition is
    - possible symptoms
    - risk level
    - precautions
    - when to see doctor

    Keep response medically responsible.
    Avoid definitive diagnosis claims.
    """

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.7
    )

    return response.choices[0].message.content