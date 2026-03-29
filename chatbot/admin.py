from django.contrib import admin
from .models import ChatMessage, Conversation, IngestedDocument
# Register your models here.
admin.site.register(ChatMessage)
admin.site.register(Conversation)
admin.site.register(IngestedDocument)
