from django.urls import path

from accountapp.views import hello_world, test, plot, hello_world_drf

#라우팅 할 수있는 환경과 로직을짜는 과정 어떤 구조로 들어갔을때 다음으로 들어갈 수 있는지
#accountapp 으로 들어가서 helloworld/로 들어가면 다음 함수를 진행 할 수 있다.
urlpatterns = [
    path('hello_world/',hello_world),
    path('test/',test),
    path('plot/', plot),

    path( 'hello_world_drf/' , hello_world_drf),

]
