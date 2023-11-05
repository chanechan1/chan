import pandas as pd
import requests
import param as pa
import matplotlib.pyplot as plt
##예측모델 예측발전량 조회
date = '2023-11-01'
bid_round = 1

weather_fcst = requests.get(f'https://research-api.solarkim.com/cmpt-2023/weathers-forecasts/{date}/{bid_round}', headers={
                            'Authorization': f'Bearer {pa.API_KEY}'
                        }).json()

weathers = requests.get(f'https://research-api.solarkim.com/open-proc/cmpt-2023/weathers-observeds/{date}', headers={
                            'Authorization': f'Bearer {pa.API_KEY}'
                        }).json()

weather_fcst=pd.DataFrame(weathers)
weathers=pd.DataFrame(weathers)

weather_fcst.plot()
weathers.plot()

plt.show()

print(weathers)
