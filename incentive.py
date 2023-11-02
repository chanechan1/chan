import requests
import json
import param as pa
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
_API_URL = "https://research-api.solarkim.com"
_API_KEY = API_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJaZlg3NGJObUNDUDhBZWI2elQ3MldoIiwiaWF0IjoxNjk4NDgxMTEzLCJleHAiOjE3MDAyMzMyMDAsInR5cGUiOiJhcGlfa2V5In0.EDbJYB23JVxxDjdn_TLBWUjq8-sV9iRVP4N8PUG3-9E'
  # https://o.solarkim.com/cmpt2023/result에서 확인할 수 있다.
_AUTH_PARAM = {"headers": {"Authorization": f"Bearer {_API_KEY}"}}


def _get(url: str):
    """
    주어진 url의 리소스를 조회한다.

    Args:
        url (str): API url
    """
    response = requests.get(url, **_AUTH_PARAM)
    return response.json()


def _post(url: str, data):
    """
    리소스 생성 데이터를 이용해서 주어진 url의 리소스를 생성한다.

    Args:
        url (str): API url
        data (dict): 리소스 생성용 데이터
    """
    response = requests.post(url, data=json.dumps(data), **_AUTH_PARAM)
    return response.json()


def _get_weathers_forecasts10():
    """
    기상데이터 일단위 기상예측 데이터 조회 (https://research-api.solarkim.com/docs#tag/Competition-2023/operation/get_weathers_forecasts_date_bid_round_cmpt_2023_weathers_forecasts__date___bid_round__get 참고)
    """
    # 오늘 날짜 구하기
    today = datetime.now()

    # 오늘 날짜에 하루 더하기
    tomorrow = today + timedelta(days=1)

    # 날짜 형식 지정 (예: '2023-10-02')
    tomorrow_formatted = tomorrow.strftime('%Y-%m-%d')

    date = tomorrow_formatted

    bid_round_10 = 1


    weather_fcst_10 = _get(
        f"{_API_URL}/cmpt-2023/weathers-forecasts/{date}/{bid_round_10}"
    )
    # 데이터프레임
    weather_fcst_10 = pd.DataFrame(weather_fcst_10)
    weather_fcst_10['time']=pd.to_datetime(weather_fcst_10['time'], utc=True)

    seoul_tz = pytz.timezone('Asia/Seoul')
    weather_fcst_10['time'] = weather_fcst_10['time'].dt.tz_convert(seoul_tz)

    print(weather_fcst_10)


    return weather_fcst_10
def _get_weathers_forecasts17():
    """
    기상데이터 일단위 기상예측 데이터 조회 (https://research-api.solarkim.com/docs#tag/Competition-2023/operation/get_weathers_forecasts_date_bid_round_cmpt_2023_weathers_forecasts__date___bid_round__get 참고)
    """
    # 오늘 날짜 구하기
    today = datetime.now()


    date = today.strftime('%Y-%m-%d')

    bid_round_17 = 2
    weather_fcst_17 = _get(
        f"{_API_URL}/cmpt-2023/weathers-forecasts/{date}/{bid_round_17}"
    )
    weather_fcst_17 = pd.DataFrame(weather_fcst_17)
    weather_fcst_17['time'] = pd.to_datetime(weather_fcst_17['time'], utc=True)

    seoul_tz = pytz.timezone('Asia/Seoul')
    weather_fcst_17['time'] = weather_fcst_17['time'].dt.tz_convert(seoul_tz)

    print(weather_fcst_17)

    return weather_fcst_17

def _get_gen_forecasts10():
    """
    더쉐어 예측 모델의 예측 발전량 조회, 입찰대상일의 5가지 예측 모델의 예측 발전량 값을 취득한다 (https://research-api.solarkim.com/docs#tag/Competition-2023/operation/get_gen_forecasts_date_cmpt_2023_gen_forecasts__date___bid_round__get 참고)
    """


    # 오늘 날짜 구하기
    today = datetime.now()

    # 오늘 날짜에 하루 더하기
    tomorrow = today + timedelta(days=1)

    # 날짜 형식 지정 (예: '2023-10-02')
    tomorrow_formatted = tomorrow.strftime('%Y-%m-%d')

    date = tomorrow_formatted

    bid_round_10 = 1

    gen_fcst_10 = _get(f"{_API_URL}/cmpt-2023/gen-forecasts/{date}/{bid_round_10}")

    #데이터프레임
    gen_fcst_10 = pd.DataFrame(gen_fcst_10)
    gen_fcst_10['time'] = pd.to_datetime(gen_fcst_10['time'], utc=True)
    seoul_tz = pytz.timezone('Asia/Seoul')
    gen_fcst_10['time'] = gen_fcst_10['time'].dt.tz_convert(seoul_tz)

    #UTC를 서울시간대로
    print(gen_fcst_10)

    return gen_fcst_10
def _get_gen_forecasts17():
    """
    더쉐어 예측 모델의 예측 발전량 조회, 입찰대상일의 5가지 예측 모델의 예측 발전량 값을 취득한다 (https://research-api.solarkim.com/docs#tag/Competition-2023/operation/get_gen_forecasts_date_cmpt_2023_gen_forecasts__date___bid_round__get 참고)
    """
    # 오늘 날짜 구하기
    today = datetime.now()
    date = today.strftime('%Y-%m-%d')
    bid_round_17 = 2

    gen_fcst_17 = _get(f"{_API_URL}/cmpt-2023/gen-forecasts/{date}/{bid_round_17}")

    # 데이터프레임
    gen_fcst_17 = pd.DataFrame(gen_fcst_17)
    gen_fcst_17['time'] = pd.to_datetime(gen_fcst_17['time'], utc=True)
    seoul_tz = pytz.timezone('Asia/Seoul')
    gen_fcst_17['time'] = gen_fcst_17['time'].dt.tz_convert(seoul_tz)

    return gen_fcst_17


def _get_weathers_observeds():
    """
    기상데이터 일단위 기상관측데이터 조회, 당일에 대해 조회하면 현재시간 기준 24시간치 조회 (https://research-api.solarkim.com/docs#tag/Competition-2023/operation/get_weathers_observeds_date_cmpt_2023_weathers_observeds__date__get 참고)
    """
    date = "2023-10-23"
    weather_obsv = _get(f"{_API_URL}/cmpt-2023/weathers-observeds/{date}")
    print(weather_obsv)


def _get_bids_result():
    """
    더쉐어 예측 모델의 예측 결과 조회 (https://research-api.solarkim.com/docs#tag/Competition-2023/operation/get_bids_result_date_cmpt_2023_bid_results__date__get 참고)
    """
    date = "2023-10-23"


    bid_results = _get(f"{_API_URL}/cmpt-2023/bid-results/{date}")
    print(bid_results)


def _post_bids(amounts):
    """
    일단위 태양광 발전량 입찰. 시간별 24개의 발전량을 입찰하며 API가 호출된 시간에 따라 입찰 대상일이 결정된다.
    """
    # NumPy 배열을 파이썬 리스트로 변환


    #success = _post(f"{_API_URL}/cmpt-2023/bids", amounts)
    success = requests.post(f'https://research-api.solarkim.com/cmpt-2023/bids', data=json.dumps(amounts), headers={
        'Authorization': f'Bearer {API_KEY}'
    }).json()

    print(amounts)
    print(success)

def _run():

    # _get_weathers_forecasts10()
    _get_weathers_observeds()
    _get_bids_result()
    print('a')
def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

if __name__ == "__main__":
    _run()
