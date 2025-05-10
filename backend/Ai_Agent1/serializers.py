from rest_framework import serializers
from .models import Scenario


class ScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scenario
        fields = ["input_text", "ai_report", "created_at"]
        read_only_fields = ["created_at"]
