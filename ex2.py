import pandas as pd
import requests
import param as pa
from datetime import datetime,timedelta
##기상데이터 일단위 예측조회
#내일 예측데이터

# 오늘 날짜 구하기
today = datetime.now()


# 오늘 날짜에 하루 더하기
tomorrow = today + timedelta(days=1)

# 날짜 형식 지정 (예: '2023-10-02')
tomorrow_formatted = tomorrow.strftime('%Y-%m-%d')


date = "2023-10-02"

bid_round = 1
weather_fcst = requests.get(f'https://research-api.solarkim.com/cmpt-2023/weathers-forecasts/{date}/{bid_round}', headers={
                            'Authorization': f'Bearer {pa.API_KEY}'
                        }).json()
weather_fcst=pd.DataFrame(weather_fcst)
print(weather_fcst)
