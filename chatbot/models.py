from django.db import models

class Conversation(models.Model):
    '''Represents a chat session'''
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'Conversation: {self.session_id}'

class ChatMessage(models.Model):
    '''Individual message in a conversation'''
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages', null=True, blank=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')
    question = models.TextField()
    answer = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    provider = models.CharField(max_length=50, default='openai')
    model = models.CharField(max_length=100, default="")
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']  # oldest first - imprtant for history order

    def __str__(self):
        return f"{self.provider} {self.question[:50]}"