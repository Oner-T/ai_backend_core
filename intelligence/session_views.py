from rest_framework.response import Response
from rest_framework.views import APIView

from intelligence.models import ChatSession, ChatMessage


class SessionListView(APIView):
    def get(self, request):
        sessions = ChatSession.objects.filter(user=request.user).values(
            'id', 'title', 'created_at', 'updated_at'
        )
        return Response(list(sessions))


class SessionMessagesView(APIView):
    def get(self, request, session_id):
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
        except ChatSession.DoesNotExist:
            return Response({'error': 'Session not found.'}, status=404)

        messages = session.messages.values('role', 'content', 'sources', 'created_at')
        return Response(list(messages))


class SessionDeleteView(APIView):
    def delete(self, request, session_id):
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
        except ChatSession.DoesNotExist:
            return Response({'error': 'Session not found.'}, status=404)

        session.delete()
        return Response(status=204)
