from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from intelligence.services import query_kvkk


class QueryView(APIView):
    def post(self, request):
        question = request.data.get("question", "").strip()
        if not question:
            return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

        history = request.data.get("history", [])
        result = query_kvkk(question, history=history)
        return Response(result, status=status.HTTP_200_OK)
