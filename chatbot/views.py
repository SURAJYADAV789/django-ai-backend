import json
import os
from openai import OpenAI
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ChatMessage 

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

SYSTEM_PROMPT = '''
You are a helpful, friendly assitant.
- Always clearly and concisely 
- If you don't know something, Say so honestly
- Format your answere in simple English
- Never make up facts
'''


@csrf_exempt
def ask_ai(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            question = data.get("question", "").strip()

            if not question:
                return JsonResponse({"error": "No question provided"}, status=400)

            # Call OpenAI (fixed: use client instead of openai)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},   # Controls AI Behavior
                    {"role": "user", "content": question}
                    ],

                temperature=0.7,   # 0 = strict/factual, 1 = creative/random
                max_tokens=500     # limit response length

            )
            answer = response.choices[0].message.content

            # Save to DB
            ChatMessage.objects.create(question=question, answer=answer)

            return JsonResponse({"question": question, "answer": answer})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST allowed"}, status=405)