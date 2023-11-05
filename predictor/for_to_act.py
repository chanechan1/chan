import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import functions as func

# 데이터 로드
forecast_data = pd.read_csv('weather_forecast.csv', parse_dates=True)
actual_data = pd.read_csv('weather_actual.csv', parse_dates=True)
# test_x = func._get_weathers_forecasts10()  ##api 내일 일기예보 가져옴
test_x = func._get_weathers_forecasts17()

# 여기에서 데이터를 전처리하고, 필요한 경우 병합합니다.
forecast_data = pd.DataFrame(forecast_data)
actual_data = pd.DataFrame(actual_data)
forecast_data = forecast_data[forecast_data.columns[2:]]
# forecast_data = forecast_data.iloc[:11568]                  #round 1
forecast_data = forecast_data.iloc[11568:]
actual_data = actual_data[actual_data.columns[1:]]

print('a')

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_forecast_data = scaler.fit_transform(forecast_data)  # 시간 열을 제외
scaled_actual_data = scaler.transform(actual_data)

# 학습 데이터셋 생성
time_steps = 10  # 예를 들어 이전 10개 시간 단계 데이터를 사용하여 예측
X, y = func.create_dataset(scaled_forecast_data, scaled_actual_data[:, 0], time_steps)  # '[:,0]'은 타겟 변수가 첫 번째 열이라고 가정

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM 모델 구축
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))  # 타겟 변수가 하나라고 가정
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), shuffle=False)

# API로부터 받은 데이터를 스케일링
test_x_processed = scaler.transform(test_x.iloc[:, 1:])  # 'iloc[:, 1:]'는 시간 열을 제외

# test_x를 시계열 데이터셋으로 변환
X_test_api, _ = func.create_dataset(test_x_processed, np.zeros((test_x_processed.shape[0],)), time_steps)

# 예측
y_pred_test_x = model.predict(X_test_api)

# 스케일 역변환 준비: placeholder를 생성합니다.
placeholder = np.zeros_like(test_x_processed)
# 예측 결과를 placeholder의 마지막 열에 넣습니다.
placeholder[:, -1] = y_pred_test_x.ravel()

# 이제 전체 배열에 inverse_transform을 적용합니다.
y_pred_test_x_rescaled = scaler.inverse_transform(placeholder)

# 마지막 열(예측 결과)만 추출합니다.
y_pred_test_x_rescaled = y_pred_test_x_rescaled[:, -1]

# 예측 결과를 DataFrame으로 변환, 시간 열을 병합합니다.
predicted_data = pd.DataFrame(y_pred_test_x_rescaled, columns=['Predicted'])
predicted_data = pd.concat([test_x.iloc[time_steps:, :1].reset_index(drop=True), predicted_data], axis=1)

print(predicted_data)
