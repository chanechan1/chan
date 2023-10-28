import pandas as pd
import numpy as np


##이전자료는 train이고 실제 할거는 test임 api를 받아와서 돌려볼 거는 test에 해당, x(다양한 변수들과)는 주어지는 발전량이고 y(실제 발전량)는 결과임
##데이터 프레임으로 변환

##여기서는 pred.csv가 y에 해당함
##gen은 실제 발전량
##pred는 예측발전량
##
train_x = pd.read_csv('weather_actual.csv', index_col=0, parse_dates=True)
train_y = pd.read_csv('weather', parse_dates=True)
test_x = pd.read_csv('testx01.xlsx', index_col=0, parse_dates=True)
test_y = pd.read_csv('testy01.xlsx', parse_dates=True)
