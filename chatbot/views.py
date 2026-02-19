from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .services import get_ai_response
from .models import ChatMessage
# Create your views here.

@api_view(['POST'])
def chat_view(request):
    message = request.data.get('message')

    if not message:
        return Response(
            {'error': 'Message is required.'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    reply = get_ai_response(message)

    # Return full chat history
    chat_history = ChatMessage.objects.all().order_by('-created_at')

    data = [
        {
            'role': chat.role,
            'content': chat.content,
            'created_at': chat.created_at
        }

        for chat in chat_history
    ]
    return Response({
        
        'reply':reply,
        'conversation': data
        
        })