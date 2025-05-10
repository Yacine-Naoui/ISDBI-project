from django.db.models.signals import pre_save
from django.dispatch import receiver
from django.core.exceptions import ValidationError
from .models import BankProfile, User


@receiver(pre_save, sender=BankProfile)
def validate_bank_domain(sender, instance, **kwargs):
    instance.clean()  # Enforce domain validation


@receiver(pre_save, sender=User)
def lowercase_email(sender, instance, **kwargs):
    instance.email = instance.email.lower()  # Ensure case-insensitive emails
