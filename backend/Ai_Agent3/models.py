from django.db import models

# Create your models here.
from users.models import User

# agent3/models.py
from django.db import models
from users.models import User


class PDFReport(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    input_pdf = models.FileField(upload_to="input_pdfs/")  # Uploaded PDF
    output_pdf = models.FileField(upload_to="output_pdfs/", null=True)  # Generated PDF
    created_at = models.DateTimeField(auto_now_add=True)
