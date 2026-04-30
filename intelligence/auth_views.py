from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken


class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email    = request.data.get('email', '').strip().lower()
        password = request.data.get('password', '')
        password2 = request.data.get('password2', '')

        if not email or not password:
            return Response({'error': 'Email and password are required.'}, status=400)
        if password != password2:
            return Response({'error': 'Passwords do not match.'}, status=400)
        if User.objects.filter(email=email).exists():
            return Response({'error': 'An account with this email already exists.'}, status=400)

        try:
            validate_password(password)
        except ValidationError as e:
            return Response({'error': ' '.join(e.messages)}, status=400)

        user = User.objects.create_user(username=email, email=email, password=password)
        refresh = RefreshToken.for_user(user)
        return Response({
            'access':  str(refresh.access_token),
            'refresh': str(refresh),
            'email':   user.email,
        }, status=201)


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email    = request.data.get('email', '').strip().lower()
        password = request.data.get('password', '')

        if not email or not password:
            return Response({'error': 'Email and password are required.'}, status=400)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({'error': 'Invalid email or password.'}, status=401)

        if not user.check_password(password):
            return Response({'error': 'Invalid email or password.'}, status=401)

        if not user.is_active:
            return Response({'error': 'Account is disabled.'}, status=401)

        refresh = RefreshToken.for_user(user)
        return Response({
            'access':  str(refresh.access_token),
            'refresh': str(refresh),
            'email':   user.email,
        })
