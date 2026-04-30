from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from intelligence.views import QueryView, FeedbackView
from intelligence.auth_views import RegisterView, LoginView
from intelligence.session_views import SessionListView, SessionMessagesView

urlpatterns = [
    # Auth
    path('auth/register/', RegisterView.as_view(),      name='register'),
    path('auth/login/',    LoginView.as_view(),         name='login'),
    path('auth/refresh/',  TokenRefreshView.as_view(),  name='token_refresh'),

    # Chat
    path('query/',    QueryView.as_view(),    name='query'),
    path('feedback/', FeedbackView.as_view(), name='feedback'),

    # Session history
    path('sessions/',                       SessionListView.as_view(),     name='sessions'),
    path('sessions/<int:session_id>/messages/', SessionMessagesView.as_view(), name='session_messages'),
]
