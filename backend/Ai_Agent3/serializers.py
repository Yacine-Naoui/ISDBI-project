# agent3/serializers.py
from rest_framework import serializers
from .models import PDFReport


class PDFReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = PDFReport
        fields = ["id", "input_pdf", "output_pdf", "created_at"]
        read_only_fields = ["id", "output_pdf", "created_at"]
