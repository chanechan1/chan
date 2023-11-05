import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import numpy as np
import incentive as it

# 데이터 로드
actual_weather = pd.read_csv('weather_actual.csv')
predictions = pd.read_csv('pred.csv')
gens = pd.read_csv('gens.csv')  # 실제 발전량 데이터를 로드

# 예측 데이터에서 'model_id'와 'round' 열을 제외하고 실측 데이터와 병합
predictions = predictions.drop(['model_id', 'round'], axis=1)
combined_data = pd.merge(actual_weather, predictions, on='time')

# 실제 발전량 데이터와 병합
combined_data_with_gens = pd.merge(combined_data, gens, on='time', how='left')

# 특성 및 타겟 분리
# amount_x 또는 amount_y 중 실제값을 나타내는 열을 사용하세요. 예를 들어, amount_y를 실제 값으로 가정합니다.
X = combined_data_with_gens.drop(['amount_x', 'amount_y', 'time'], axis=1)
y = combined_data_with_gens['amount_y']  # 실제 발전량을 나타내는 열을 타겟 변수로 사용
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# 데이터 분할
X = combined_data_with_gens.drop(['amount_x', 'amount_y', 'time'], axis=1)
y = combined_data_with_gens['amount_y']

X_values = X.values
X_scaled = scaler_x.fit_transform(X_values)

y_values = y.values.reshape(-1, 1)  # y가 1D 배열인 경우 2D 배열로 변환
y_scaled = scaler_y.fit_transform(y_values)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# LSTM을 위한 데이터 재구조화
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# LSTM 모델 생성
input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
model = Sequential()
model.add(LSTM(50, input_shape=input_shape))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# 모델 훈련
history = model.fit(X_train_reshaped, y_train, epochs=1, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=2, shuffle=False)

# 예측
y_pred_scaled = model.predict(X_test_reshaped)

# 스케일링 되돌리기
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

# 예측 결과 시각화
plt.figure(figsize=(15, 6))
plt.plot(y_test_original, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("Energy Generation Prediction")
plt.xticks(range(len(y_pred)), range(len(y_pred)))
plt.grid(True)
plt.show()