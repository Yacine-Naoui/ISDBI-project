from rest_framework import serializers
from .models import User, BankProfile


class BankProfileSerializer(serializers.ModelSerializer):
    license_file = serializers.FileField(required=True)
    bank_logo = serializers.ImageField(required=False)

    class Meta:
        model = BankProfile
        fields = [
            "bank_name",
            "official_email_domain",
            "license_number",
            "license_file",
            "bank_logo",
        ]


# users/serializers.py
class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    bank_profile = BankProfileSerializer()

    class Meta:
        model = User
        fields = [
            "email",
            "username",
            "password",
            "is_verified",
            "is_bank",
            "bank_profile",
        ]
        read_only_fields = ["is_verified", "is_bank"]

    def create(self, validated_data):
        bank_profile_data = validated_data.pop("bank_profile")
        user = User.objects.create_user(
            email=validated_data["email"],
            password=validated_data["password"],
            is_bank=True,
            username=validated_data.get("username", None),  # Optional username
        )
        BankProfile.objects.create(user=user, **bank_profile_data)
        return user
