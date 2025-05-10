# ai/urls.py
from django.urls import path
from .views import ProcessScenarioView

urlpatterns = [
    path("process-scenario/", ProcessScenarioView.as_view(), name="process-scenario"),
]
