from rest_framework import viewsets

from patternanalysis.drf.serializers import AiUserListSerializer
from patternanalysis.models import AiUserList


class AiUserViewSet(viewsets.ModelViewSet):
    queryset = AiUserList.objects.all().order_by('name')
    serializer_class = AiUserListSerializer
