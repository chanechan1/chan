# import pandas as pd
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
# # 상관계수 계산
# correlation_matrix_actual = actual_weather_data.corr()
# correlation_matrix_predicted = predicted_weather_data.corr()
#
# # 각 열의 가중치 계산
# cloud_weight = correlation_matrix_actual['cloud']['cloud']
# temp_weight = correlation_matrix_actual['temp']['temp']
# humidity_weight = correlation_matrix_actual['humidity']['humidity']
# ground_press_weight = correlation_matrix_actual['ground_press']['ground_press']
# wind_speed_weight = correlation_matrix_actual['wind_speed']['wind_speed']
# wind_dir_weight = correlation_matrix_actual['wind_dir']['wind_dir']
# rain_weight = correlation_matrix_actual['rain']['rain']
# snow_weight = correlation_matrix_actual['snow']['snow']
# dew_point_weight = correlation_matrix_actual['dew_point']['dew_point']
# vis_weight = correlation_matrix_actual['vis']['vis']
# uv_idx_weight = correlation_matrix_actual['uv_idx']['uv_idx']
# azimuth_weight = correlation_matrix_actual['azimuth']['azimuth']
# elevation_weight = correlation_matrix_actual['elevation']['elevation']
#
# # 각 데이터 프레임에 가중 평균 계산
# actual_weather_data['weighted_combined'] = (
#     actual_weather_data['cloud'] * cloud_weight +
#     actual_weather_data['temp'] * temp_weight +
#     actual_weather_data['humidity'] * humidity_weight +
#     actual_weather_data['ground_press'] * ground_press_weight +
#     actual_weather_data['wind_speed'] * wind_speed_weight +
#     actual_weather_data['wind_dir'] * wind_dir_weight +
#     actual_weather_data['rain'] * rain_weight +
#     actual_weather_data['snow'] * snow_weight +
#     actual_weather_data['dew_point'] * dew_point_weight +
#     actual_weather_data['vis'] * vis_weight +
#     actual_weather_data['uv_idx'] * uv_idx_weight +
#     actual_weather_data['azimuth'] * azimuth_weight +
#     actual_weather_data['elevation'] * elevation_weight
# )
#
# predicted_weather_data['weighted_combined'] = (
#     predicted_weather_data['cloud'] * cloud_weight +
#     predicted_weather_data['temp'] * temp_weight +
#     predicted_weather_data['humidity'] * humidity_weight +
#     predicted_weather_data['ground_press'] * ground_press_weight +
#     predicted_weather_data['wind_speed'] * wind_speed_weight +
#     predicted_weather_data['wind_dir'] * wind_dir_weight +
#     predicted_weather_data['rain'] * rain_weight +
#     predicted_weather_data['snow'] * snow_weight +
#     predicted_weather_data['dew_point'] * dew_point_weight +
#     predicted_weather_data['vis'] * vis_weight +
#     predicted_weather_data['uv_idx'] * uv_idx_weight +
#     predicted_weather_data['azimuth'] * azimuth_weight +
#     predicted_weather_data['elevation'] * elevation_weight
# )
#
# # 결과 데이터를 합칠 수 있음
# result_data = pd.concat([actual_weather_data, predicted_weather_data], axis=1)
#
# # 가중 평균 값 출력
# print(result_data[['time', 'weighted_combined']])
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

model.fit(train_x_actual_reshaped, train_y_actual_scaled, epochs=1000, batch_size=64, verbose=2, shuffle=False)
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
prediction_ratio = yhat_original / train_y_actual_values

# 비율을 시각화
plt.figure(figsize=(15, 6))
plt.plot(prediction_ratio, label='Prediction Ratio')
plt.axhline(y=1, color='r', linestyle='--', label='Perfect Prediction (Ratio = 1)')
plt.legend()
plt.title("Prediction Ratio")
plt.show()
