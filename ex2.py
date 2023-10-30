import requests
import param as pa

##기상데이터 일단위 예측조회
date = '2023-10-31'
bid_round = 2
weather_fcst = requests.get(f'https://research-api.solarkim.com/cmpt-2023/weathers-forecasts/{date}/{bid_round}', headers={
                            'Authorization': f'Bearer {pa.API_KEY}'
                        }).json()
print(weather_fcst)
