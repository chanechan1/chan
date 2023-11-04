# import pandas as pd
# import numpy as np
#
# # 데이터 로드
# actual_weather_data = pd.read_csv('weather_actual.csv')
# predicted_weather_data = pd.read_csv('weather_forecast.csv')
# predicted_weather_data = predicted_weather_data[predicted_weather_data.columns[1:]]
# predicted_weather_data = predicted_weather_data.iloc[:11616]
#
# # 데이터 전처리: 필요한 열 선택
# actual_weather_data = actual_weather_data[['time', 'cloud', 'temp', 'humidity', 'ground_press', 'wind_speed', 'wind_dir', 'rain', 'snow', 'dew_point', 'vis', 'uv_idx', 'azimuth', 'elevation']]
# predicted_weather_data = predicted_weather_data[['cloud', 'temp', 'humidity', 'ground_press', 'wind_speed', 'wind_dir', 'rain', 'snow', 'dew_point', 'vis', 'uv_idx', 'azimuth', 'elevation']]
#
# # 각 열에 대한 오차율 계산
# for column in actual_weather_data.columns[1:]:
#     actual_col_name = column
#     predicted_col_name = column
#     actual_weather_data[f'{column}_오차율(%)'] = ((predicted_weather_data[predicted_col_name]) - actual_weather_data[actual_col_name]  / actual_weather_data[actual_col_name]) * 100
#
#
# # 결과 데이터 출력
# print(actual_weather_data)

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# 실제 기상 정보와 실제 발전량 데이터 불러오기
train_x_actual = pd.read_csv('weather_actual.csv', parse_dates=True)
train_y_actual = pd.read_csv('gens.csv', parse_dates=True)

# 예측 기상 정보와 실제 발전량 데이터 불러오기
train_x_predicted = pd.read_csv('weather_forecast.csv', parse_dates=True)
train_x_predicted = train_x_predicted[train_x_predicted.columns[1:]]
train_x_predicted = train_x_predicted.iloc[:11616]
train_y_actual = pd.read_csv('gens.csv', parse_dates=True)

# 데이터 전처리: 필요한 열 선택 및 정규화
train_x_actual = train_x_actual[train_x_actual.columns[1:]]
train_y_actual = train_y_actual[train_y_actual.columns[1:]]
train_x_predicted = train_x_predicted[train_x_predicted.columns[1:]]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

train_x_actual_values = train_x_actual.values
train_x_actual_scaled = scaler_x.fit_transform(train_x_actual_values)
train_y_actual_values = train_y_actual.values
train_y_actual_scaled = scaler_y.fit_transform(train_y_actual_values)

train_x_predicted_values = train_x_predicted.values
train_x_predicted_scaled = scaler_x.transform(train_x_predicted_values)

train_x_actual_reshaped = np.expand_dims(train_x_actual_scaled, axis=1)
train_x_predicted_reshaped = np.expand_dims(train_x_predicted_scaled, axis=1)



input_shape = (1, train_x_actual_reshaped.shape[2])

model = Sequential()
model.add(LSTM(50, input_shape=input_shape))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(train_x_actual_reshaped, train_y_actual_scaled, epochs=100, batch_size=32, verbose=2, shuffle=False)
# 모델 예측
train_x_predicted_reshaped = train_x_predicted_scaled.reshape((train_x_predicted_scaled.shape[0], 1, train_x_predicted_scaled.shape[1]))
yhat = model.predict(train_x_predicted_reshaped)

# 역 정규화
yhat_original = scaler_y.inverse_transform(yhat)

# 결과 출력
plt.figure(figsize=(15, 6))
plt.plot(train_y_actual_values, label='Actual')
plt.plot(yhat_original, label='Predicted')
plt.legend()
plt.show()

# 예측된 발전량과 실제 발전량의 비율 계산
prediction_ratio = train_y_actual_values/yhat_original
prediction_ratio[np.isinf(prediction_ratio)] = 0 # inf값 전부 0으로 날려버림(의미없음 어차피)
prediction_ratio[np.abs(prediction_ratio) >= 20] = 0

split_ratio = np.array_split(prediction_ratio, 24)
final_weight = pd.DataFrame({'Hour': list(range(24)), 'Average_Ratio': [np.abs.mean(x) for x in split_ratio]})

# 비율을 시각화
plt.figure(figsize=(15, 6))
plt.plot(prediction_ratio, label='Prediction Ratio')
plt.axhline(y=1, color='r', linestyle='--', label='Perfect Prediction (Ratio = 1)')
plt.legend()
plt.title("Prediction Ratio")
plt.show()
print(final_weight)
