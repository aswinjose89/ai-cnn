from rest_framework import serializers

from patternanalysis.models import AiUserList

class AiUserListSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = AiUserList
        fields = ('name', 'alias')
