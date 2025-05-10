from django.db import models

# Create your models here.
from users.models import User


# Create your models here.
class Scenario(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    scenario_name = models.CharField(max_length=255)
    input_text = models.TextField()
    ai_report = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.scenario_name
