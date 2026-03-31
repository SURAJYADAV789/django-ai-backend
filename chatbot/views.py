import json
import os
from openai import OpenAI
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ChatMessage, Conversation, IngestedDocument
from django_ratelimit.decorators import ratelimit
from django.views.decorators.http import require_POST, require_GET
from .ai_providers.router import get_provider
from chatbot.rag.rag_pipeline import ask_with_rag
from .rag.vector_store import semantic_search

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

SYSTEM_PROMPT = '''
You are a helpful, friendly assistant.
- Always respond clearly and concisely 
- If you don't know something, Say so honestly
- Format your answeres in simple English
- Never make up facts
- Remember what the user tells you during the conversations
'''

MAX_HISTORY = 10

def build_messages(conversation, system_prompt, current_question):
    '''
    Build the full messages list to send to API.
    Includes system prompt + last N messages + current questions.
    '''
    messages = [{'role': 'system', 'content': system_prompt}]

    # Load last MAx_HISTORY messages from DB
    history = conversation.messages.order_by('-created_at')[:MAX_HISTORY]
    history = reversed(list(history))  # oldest first

    for msg in history:
        messages.append({'role': 'user', 'content': msg.question})
        messages.append({'role': 'assistant', 'content': msg.answer})

    messages.append({'role': 'user', 'content': current_question})

    return messages


@csrf_exempt
@require_POST
@ratelimit(key='ip', rate='10/m', block=True)  # Max 10 requests per minute per ip
def ask_ai(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            question = data.get("question", "").strip()
            session_id = data.get("session_id", "").strip()  # <- New

            if not question:
                return JsonResponse({"error": "No question provided"}, status=400)
            
            if not session_id:
                return JsonResponse({'error': 'No session_id provided'}, status=400)
            
            # Get or create conversation for this session
            conversation, created = Conversation.objects.get_or_create(
                session_id=session_id
            )
            
            # Build full messsage history
            provider = get_provider()
            result = provider.complete(question, SYSTEM_PROMPT)
            messages = build_messages(conversation, SYSTEM_PROMPT, question)

            # call AI with full history
            result = provider.complete_with_messages(messages)



            # Save to DB
            ChatMessage.objects.create(
                conversation=conversation,
                question=question, 
                answer=result.answer,
                ip_address=request.META.get('REMOTE_ADDR'),
                provider=result.provider,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                )

            return JsonResponse({"question": question, "answer": result.answer, "session_id": session_id})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST allowed"}, status=405)


@require_GET
def get_history(request, session_id):
    '''Return full conversation history for a session'''
    try:
        conversation = Conversation.objects.get(session_id=session_id)
        messages = list(conversation.messages.all().values(
            'question', 'answer', 'created_at', 'provider'
        ))

        # convert datetime to string fro json serialization
        for msg in messages:
            msg['created_at'] = msg['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            
        return JsonResponse({'session_id': session_id, 'messages': list(messages)})
    except Conversation.DoesNotExist:
        return JsonResponse({'error': 'Session not found'}, status=400)
    

@csrf_exempt
@require_POST
@ratelimit(key='ip', rate='10/m', block=True)
def rag_ask(request):
    '''
    RAG ENDPOINT - Answere Questions from your INGESTED documents

    Differnce from /chat/:
    /chat/ -> answers from GPT - 4o genernal knowledge
    /rag/  -> answere ONLY from your ingested documents
    '''
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        session_id = data.get('session_id', '').strip()

        if not question:
            return JsonResponse({'error': 'No question provides'}, status=400)
        
        if not session_id:
            return JsonResponse({'error': 'No session_id provided'}, status=400)
        
        # check if any documents have been ingestes
        if not IngestedDocument.objects.exists():
            return JsonResponse({
                'error': 'No documents ingested yet. Run python manage.py ingest_docs --file <path>'
            },status=400)
        
        # get or create conversation for memory 
        conversation, created = Conversation.objects.get_or_create(
            session_id=f'rag_{session_id}'  # Prefix to separate from regular chat
        )

        # load last 5 message from context
        history = []
        past_messages = conversation.messages.order_by('-created_at')[:5]
        for msg in reversed(list(past_messages)):
            history.append({
                'question': msg.question,
                'answer': msg.answer
            })


        # Run RAG pipeline
        result = ask_with_rag(
            question=question,
            n_chunks=3,
            conversation_history=history if history else None,
        )


        # save to DB
        ChatMessage.objects.create(
            conversation=conversation,
            question=question,
            answer=result.answer,
            ip_address=request.META.get('REMOTE_ADDR'),
            provider=result.provider,
            model=result.model, 
            input_tokens=0,   # RAG use variable token
            output_tokens=0
        )

        return JsonResponse({
            'question': question,
            'answer': result.answer,
            'sources': result.sources,   # which docs where used
            'session_id': session_id,
        })
        
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid Json'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    

@require_GET
def list_documents(request):
    '''Show what documents have been ingested'''
    docs = IngestedDocument.objects.all().values(
        'filename', 'chunk_count', 'ingested_at'
    )

    docs_list = list(docs)
    for doc in docs_list:
        doc['ingested_at'] = doc['ingested_at'].strftime('%Y-%m-%d %H:%M')

    return JsonResponse({
        'total_documents': len(docs_list),
        'documents': docs_list,
    })


@csrf_exempt
@require_POST
@ratelimit(key='ip', rate='20/m', block=True)
def search_documents(request):
    """
    semantic search endpoint
    searches your ingested documents by meaning, not keywords

    """

    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()
        n_results = data.get("n_results", 5)
        min_similarity = data.get('min_similarity', 0.3)

        if not query:
            return JsonResponse({'error': 'No query provided'}, status= 400)
        
        # check documents exists
        if not IngestedDocument.objects.exists():
            return JsonResponse({
                "error": "No documents ingested yet.",
                "hint": "Run: python manage.py ingest_docs --file <path>"
            }, status=400)
        
        # Run semantic search
        results = semantic_search(
            query=query,
            n_results=n_results,
            min_similarity=min_similarity
        )

        return JsonResponse({
            "query": query,
            "total_results": len(results),
            "results": results
        })
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
        
