from django.urls import path
from .views import ask_ai, get_history

urlpatterns = [
    path('chat/',ask_ai),
    path('history/<str:session_id>/',get_history),
]