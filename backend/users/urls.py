from django.urls import path
from .views import BankView

urlpatterns = [
    path("bank/", BankView.as_view(), name="bank-register"),
]
