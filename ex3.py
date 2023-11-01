import pandas as pd
import requests
import param as pa
date = '2023-10-27'
bid_results = requests.get(f'https://research-api.solarkim.com/open-proc/cmpt-2023/bid-results/{date}', headers={
                            'Authorization': f'Bearer {pa.API_KEY}'
                        }).json()
print(bid_results)
