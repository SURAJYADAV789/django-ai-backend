from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .services import get_ai_response

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
    return Response({'reply':reply})