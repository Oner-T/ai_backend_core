from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from intelligence.services import query_kvkk
from intelligence.models import UserFeedback, ChatSession, ChatMessage


class QueryView(APIView):
    def post(self, request):
        question = request.data.get('question', '').strip()
        if not question:
            return Response({'error': 'Question is required.'}, status=status.HTTP_400_BAD_REQUEST)

        history    = request.data.get('history', [])
        session_id = request.data.get('session_id')
        regime     = request.data.get('regime', 'tr')
        if regime not in ('tr', 'eu'):
            regime = 'tr'

        result = query_kvkk(question, history=history, regime=regime)

        # ── Persist to chat session ───────────────────────────────────────────
        session = None
        if session_id:
            try:
                session = ChatSession.objects.get(id=session_id, user=request.user)
            except ChatSession.DoesNotExist:
                pass

        if session is None:
            session = ChatSession.objects.create(
                user=request.user,
                title=question[:100],
            )

        ChatMessage.objects.create(session=session, role='user',      content=question)
        ChatMessage.objects.create(session=session, role='assistant', content=result['answer'], sources=result.get('sources', []))

        return Response({**result, 'session_id': session.id}, status=status.HTTP_200_OK)


class FeedbackView(APIView):
    def post(self, request):
        question = request.data.get('question', '').strip()
        answer   = request.data.get('answer', '').strip()
        rating   = request.data.get('rating', '').strip()

        if not question or not answer or rating not in ('good', 'bad'):
            return Response({'error': 'question, answer, and rating (good|bad) are required.'},
                            status=status.HTTP_400_BAD_REQUEST)

        UserFeedback.objects.create(
            question=question,
            answer=answer,
            sources=request.data.get('sources', []),
            rating=rating,
            comment=request.data.get('comment', ''),
        )
        return Response({'status': 'saved'}, status=status.HTTP_201_CREATED)
