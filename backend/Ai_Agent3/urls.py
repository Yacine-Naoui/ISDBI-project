from django.urls import path
from .views import PDFProcessingView

urlpatterns = [
    path("process-pdf/", PDFProcessingView.as_view(), name="process_pdf"),
]
