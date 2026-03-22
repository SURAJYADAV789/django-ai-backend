from django.db import models

class ChatMessage(models.Model):
    question = models.TextField()
    answer = models.TextField()
    ip_address =models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.question[:50]