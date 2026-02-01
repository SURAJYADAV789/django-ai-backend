import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_ai_response(user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": "You are a helpful assitant."},
            {'role': "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content
    