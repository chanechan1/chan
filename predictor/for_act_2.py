import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import functions as func

# 데이터 로드
forecast_data = pd.read_csv('weather_forecast.csv')
actual_data = pd.read_csv('weather_actual.csv')
test_x = func._get_weathers_forecasts10()  ## API로부터 내일의 일기예보를 가져옴
#test_x = func._get_weathers_forecasts17()  ## API로부터 일기예보를 가져옴

# 여기에서 데이터를 전처리하고, 필요한 경우 병합합니다.
forecast_data = pd.DataFrame(forecast_data)
actual_data = pd.DataFrame(actual_data)
forecast_data = forecast_data.iloc[:, 2:]  # 시간 열을 제외
forecast_data = forecast_data.iloc[:11568]
#forecast_data = forecast_data.iloc[11568:]
actual_data = actual_data.iloc[:, 1:]

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_forecast_data = scaler.fit_transform(forecast_data)
scaled_actual_data = scaler.transform(actual_data)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(scaled_forecast_data, scaled_actual_data, test_size=0.2, random_state=42)

# 완전 연결 신경망 모델 구축
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1]))  # 출력 레이어는 타겟 변수의 수와 일치해야 합니다. 여기서는 scaled_actual_data의 열 수입니다.
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# API로부터 받은 데이터를 스케일링
test_x_processed = scaler.transform(test_x.iloc[:, 1:])  # 'iloc[:, 2:]'는 시간 열을 제외

# 예측
y_pred_test_x = model.predict(test_x_processed)

# 스케일 역변환
y_pred_test_x_rescaled = scaler.inverse_transform(y_pred_test_x)

# 예측 결과를 DataFrame으로 변환, 시간 열을 병합합니다.
predicted_data = pd.DataFrame(y_pred_test_x_rescaled, columns=actual_data.columns[0:])  # 컬럼 이름을 actual_data에서 가져옵니다.
predicted_data = pd.concat([test_x.iloc[:, :1].reset_index(drop=True), predicted_data], axis=1)  # 'iloc[:, :2]'는 시간과 다른 식별 열을 가져옵니다.
for column in ['cloud', 'rain', 'snow']:
    predicted_data[column] = np.where(test_x[column] == 0, 0, predicted_data[column])

predicted_data['cloud'] = predicted_data['cloud'].clip(lower=0, upper=100)
predicted_data['uv_idx'] = predicted_data['uv_idx'].clip(lower=0)
print(predicted_data)
print('a')
