import os
from openai import OpenAI
from .models import ChatMessage


client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_ai_response(user_message: str) -> str:
    
    # Save Message
    ChatMessage.objects.create(role='user', content=user_message)

    # Fetch full conversation history
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}
    ]

    chat_history = ChatMessage.objects.all().order_by('-created_at')

    for chat in chat_history:
        messages.append({
            'role': chat.role,
            'content': chat.content
        })

    # Send to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )



    ai_reply = response.choices[0].message.content

    # Save assistant reply
    ChatMessage.objects.create(role='assistant', content=ai_reply)

    return ai_reply
    