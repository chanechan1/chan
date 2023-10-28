import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


##이전자료는 train이고 실제 할거는 test임 api를 받아와서 돌려볼 거는 test에 해당, x(다양한 변수들과)는 주어지는 발전량이고 y(실제 발전량)는 결과임
##데이터 프레임으로 변환

##여기서는 pred.csv가 y에 해당함
##gen은 실제 발전량
##pred는 예측발전량
##
train_x = pd.read_csv('weather_actual.csv', index_col=0, parse_dates=True)
train_y = pd.read_csv('gens.csv',parse_dates=True)
test_x = pd.read_csv('weather_forecast.csv', index_col=1, parse_dates=True)
test_y = pd.read_csv('pred.csv', parse_dates=True)

##데이터 재정렬
train_x=train_x[train_x.columns[1:]] #날씨정보
train_y=train_y[train_y.columns[1:]] #발전정보


test_x=test_x[test_x.columns[2:]] #1에 대한 일기예보만 살리기위해 일단 없앰
test_x=test_x.iloc[:11616]        #1에 대한 것만 살림

test_y=test_y.iloc[:58080]
test_y=test_y[test_y.columns[3:]] #amount만 남기고
test_y=test_y.iloc[2::5]

#정규화
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

##값들만 추출
train_x_values = train_x.values
train_x_scaled = scaler_x.fit_transform(train_x_values)

train_y_values = train_y.values
train_y_scaled = scaler_y.fit_transform(train_y_values)

test_x_values = test_x.values
test_x_scaled = scaler_x.transform(test_x_values)

test_y_values = test_y.values
test_y_scaled = scaler_y.transform(test_y_values)


train_x_reshaped = train_x_scaled.reshape((train_x_scaled.shape[0], 1, train_x_scaled.shape[1]))
test_x_reshaped = test_x_scaled.reshape((test_x_scaled.shape[0], 1, test_x_scaled.shape[1]))

#그냥 해보기
input_shape = (1, train_x_scaled.shape[1])

model = Sequential()
model.add(LSTM(50, input_shape=input_shape))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


history = model.fit(train_x_reshaped, train_y_scaled, epochs=100, batch_size=32, validation_data=(test_x_reshaped, test_y_scaled), verbose=2, shuffle=False)

yhat = model.predict(test_x_reshaped)

yhat_original = scaler_y.inverse_transform(yhat)
test_y_original = scaler_y.inverse_transform(test_y_scaled)

plt.figure(figsize=(15,6))
plt.plot(test_y_original, label='Actual')
plt.plot(yhat_original, label='Predicted')
plt.legend()
plt.show()



