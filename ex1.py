import requests
import param as pa
##예측모델 예측발전량 조회
date = '2023-10-23'
bid_round = 1
gen_fcst = requests.get(f'https://research-api.solarkim.com/cmpt-2023/gen-forecasts/{date}/{bid_round}', headers={
                            'Authorization': f'Bearer {pa.API_KEY}'
                        }).json()
print(gen_fcst)
print('a')
