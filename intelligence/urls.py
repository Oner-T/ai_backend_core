from django.urls import path
from intelligence.views import QueryView, FeedbackView

urlpatterns = [
    path("query/", QueryView.as_view(), name="query"),
    path("feedback/", FeedbackView.as_view(), name="feedback"),
]
