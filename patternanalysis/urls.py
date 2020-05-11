from django.urls import include, path
from rest_framework import routers
from . import views
from .drf import viewsets

router = routers.DefaultRouter()
router.register(r'aiUserList', viewsets.AiUserViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('image-processor', views.ImageAnalyzerView.as_view(), name="image_processor"),
    path('neurons', views.ModelNeuronView.as_view(), name="neurons"),
    path('articles', views.ArticleView.as_view(), name="articles")
]
