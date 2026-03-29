from django.urls import path
from .views import ask_ai, get_history, rag_ask, list_documents

urlpatterns = [
    path('chat/',ask_ai),   # regular chatbot   
    path('history/<str:session_id>/',get_history),  # conversation history
    path('rag/', rag_ask),               # RAG chatbot ← new
    path('rag/documents/', list_documents),  # list ingested docs ← new
]