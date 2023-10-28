import requests
import param as pa


date = '2023-10-01'
weathers = requests.get(f'https://research-api.solarkim.com/open-proc/cmpt-2023/weathers-observeds/{date}', headers={
                            'Authorization': f'Bearer {pa.API_KEY}'
                        }).json()
print(weathers)
