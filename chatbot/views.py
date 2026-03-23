import json
import os
from openai import OpenAI
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ChatMessage 
from django_ratelimit.decorators import ratelimit
from django.views.decorators.http import require_POST
from .ai_providers.router import get_provider

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

SYSTEM_PROMPT = '''
You are a helpful, friendly assistant.
- Always respond clearly and concisely 
- If you don't know something, Say so honestly
- Format your answeres in simple English
- Never make up facts
'''


@csrf_exempt
@require_POST
@ratelimit(key='ip', rate='10/m', block=True)  # Max 10 requests per minute per ip
def ask_ai(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            question = data.get("question", "").strip()

            if not question:
                return JsonResponse({"error": "No question provided"}, status=400)
            
            provider = get_provider()
            result = provider.complete(question, SYSTEM_PROMPT)


            # Save to DB
            ChatMessage.objects.create(
                question=question, 
                answer=result.answer,
                ip_address=request.META.get('REMOTE_ADDR'),
                provider=result.provider,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                )

            return JsonResponse({"question": question, "answer": result.answer})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST allowed"}, status=405)