import io
import matplotlib.pyplot as plt
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response

import predictor
# Create your views here.

#기존 장고 방식
def hello_world(request):
    return HttpResponse('hi')

def test(request):
    a=pd.read_csv('C:\\Users\\ckk\\Documents\\ssproject\\predictor\\gens.csv')
    a=a.to_html()
    return HttpResponse(a)


def plot(request):
    # 데이터 로드 및 numpy 배열로 변환
    a = pd.read_csv('C:\\Users\\ckk\\Documents\\ssproject\\predictor\\gens_by_sugi.csv')

    # 'time' 열을 날짜/시간 객체로 변환
    a['time'] = pd.to_datetime(a['time'])

    # 그래프 생성
    plt.figure(figsize=(15, 6))
    plt.plot(a['time'], a['amount'], label='amount')  # 'time'을 x축, 'amount'를 y축으로 사용

    # 버퍼 생성 및 그래프 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    # 버퍼의 내용을 HTTP 응답으로 반환
    return HttpResponse(buf.getvalue(), content_type='image/png')

#DRF 레스트프레임워크 를 통한 방식 과제 이걸로 할 거임

@api_view() #데코레이터
def hello_world_drf(request):
    return Response({"message":"i'm kwon byeonchan"})
