from django.db import models

# Create your models here.

class ChatMessage(models.Model):
    ROLE_CHOICES = (
        ('user','User'),
        ('assitant','assitant'),
    )

    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.role}: {self.content[:30]}"
    


