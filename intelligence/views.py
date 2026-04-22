from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from intelligence.services import query_kvkk
from intelligence.models import UserFeedback


class QueryView(APIView):
    def post(self, request):
        question = request.data.get("question", "").strip()
        if not question:
            return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

        history = request.data.get("history", [])
        result = query_kvkk(question, history=history)
        return Response(result, status=status.HTTP_200_OK)


class FeedbackView(APIView):
    def post(self, request):
        question = request.data.get("question", "").strip()
        answer   = request.data.get("answer", "").strip()
        rating   = request.data.get("rating", "").strip()

        if not question or not answer or rating not in ("good", "bad"):
            return Response({"error": "question, answer, and rating (good|bad) are required."},
                            status=status.HTTP_400_BAD_REQUEST)

        UserFeedback.objects.create(
            question = question,
            answer   = answer,
            sources  = request.data.get("sources", []),
            rating   = rating,
            comment  = request.data.get("comment", ""),
        )
        return Response({"status": "saved"}, status=status.HTTP_201_CREATED)
