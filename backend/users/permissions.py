from rest_framework.permissions import BasePermission, SAFE_METHODS


class IsVerifiedBankUser(BasePermission):
    """
    Custom permission to only allow verified bank users to access certain views.
    """

    def has_permission(self, request, view):

        # Check if the user is authenticated and is a verified bank user
        return (
            request.user.is_authenticated
            and request.user.is_verified
            and request.user.is_bank
        )

