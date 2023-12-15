from django.urls import path
from . import views
from .views import get_weather_data

app_name = 'weather'

urlpatterns = [
    #path('get_weather_data/', get_weather_data, name='get_weather_data'),
    path('', views.index, name='index'),  # pred.html을 기본 페이지로 설정
    path('get_weather_data/', views.get_weather_data, name='get_weather_data'), # 예측값 가져오는 경로
]
