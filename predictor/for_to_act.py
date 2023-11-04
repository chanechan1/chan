import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 데이터 로드
forecast_data = pd.read_csv('weather_forecast.csv', parse_dates=True)
actual_data = pd.read_csv('weather_actual.csv', parse_dates=True)

# 여기에서 데이터를 전처리하고, 필요한 경우 병합합니다.
forecast_data = pd.DataFrame(forecast_data)
actual_data = pd.DataFrame(actual_data)
forecast_data = forecast_data[forecast_data.columns[2:]]
forecast_data = forecast_data.iloc[:11616]
actual_data = actual_data[actual_data.columns[1:]]


# LSTM에 넣을 수 있도록 데이터를 변환하는 함수
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# 정규화를 위한 스케일러 정의
scaler = MinMaxScaler(feature_range=(0, 1))

# 예측하고자 하는 타겟 변수를 정규화합니다.
target_variable = 'cloud'
forecast_data[target_variable] = scaler.fit_transform(forecast_data[[target_variable]])

# 시계열 데이터 생성
time_steps = 10
X, y = create_dataset(forecast_data, forecast_data[target_variable], time_steps)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


# 모델 훈련
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    shuffle=False
)

# 예측
y_pred = model.predict(X_test)

# 실제값으로 스케일 역변환
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred)

plt.figure(figsize=(15, 5))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Weather Forecast vs Actual Data')
plt.xlabel('Time Step')
plt.ylabel('Temperature')
plt.legend()
plt.show()
# 성능 평가
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
