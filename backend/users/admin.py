from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, BankProfile


class BankProfileInline(admin.StackedInline):
    model = BankProfile
    fields = ["bank_name", "official_email_domain", "license_number", "logo"]
    readonly_fields = ["official_email_domain"]


class CustomUserAdmin(UserAdmin):
    list_display = ("email", "is_bank", "is_verified")
    ordering = ["email"]
    inlines = [BankProfileInline]
    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Permissions", {"fields": ("is_bank", "is_verified", "is_staff")}),
    )


admin.site.register(User, CustomUserAdmin)
