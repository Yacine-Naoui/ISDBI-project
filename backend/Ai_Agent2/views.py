from django.shortcuts import render
from CreateEmbedding import vectore_store
from agent2 import analyze_transaction

# ai/views.py
from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework import status
from .models import Scenario

# Modified to stream output
from users.permissions import IsVerifiedBankUser


class ProcessScenarioView(APIView):
    permission_classes = [IsVerifiedBankUser]

    def post(self, request):
        input_text = request.data.get("input_text")
        user = request.user

        def stream_generator():
            full_report = []
            # Stream AI response chunk-by-chunk
            result = analyze_transaction(vectore_store, input_text)
            for chunk in result:
                full_report.append(chunk)
                yield chunk  # Send to cliecent immediately

            # After streaming completes, save full report
            Scenario.objects.create(
                user=user, input_text=input_text, ai_report="".join(full_report)
            )

        return StreamingHttpResponse(stream_generator(), content_type="text/plain")


# Create your views here.
