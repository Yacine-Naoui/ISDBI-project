from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError
from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("The Email field must be set")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user


class User(AbstractUser):
    # Remove username and use email for authentication
    username = models.CharField(max_length=30, blank=True, null=True)
    email = models.EmailField(unique=True)
    is_bank = models.BooleanField(default=False)
    is_verified = models.BooleanField(default=False)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    def __str__(self):
        return self.email


class BankProfile(models.Model):
    user = models.OneToOneField(
        User, on_delete=models.CASCADE, related_name="bank_profile"
    )
    bank_name = models.CharField(max_length=100, unique=True)
    official_email_domain = models.CharField(max_length=100)  # Not EmailField!
    license_number = models.CharField(max_length=255, unique=True)
    license_file = models.FileField(upload_to="bank_licenses/")

    bank_logo = models.ImageField(
        upload_to="bank_logos/",
        validators=[FileExtensionValidator(allowed_extensions=["jpg", "jpeg", "png"])],
        blank=True,
        null=True,
    )

    def __str__(self):
        return self.bank_name

    def clean(self):
        # Validate email domain matches official domain
        user_domain = self.user.email.split("@")[-1].lower()
        official_domain = self.official_email_domain.lower().strip()
        if user_domain != official_domain:
            raise ValidationError(
                f"User email domain '{user_domain}' does not match official domain '{official_domain}'."
            )

    def save(self, *args, **kwargs):
        self.full_clean()  # Enforce validation on save
        super().save(*args, **kwargs)
