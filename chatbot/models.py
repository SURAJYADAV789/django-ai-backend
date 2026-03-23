from django.db import models

class ChatMessage(models.Model):
    question = models.TextField()
    answer = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    provider = models.CharField(max_length=50, default='openai')
    model = models.CharField(max_length=100, default="")
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.provider} {self.question[:50]}"