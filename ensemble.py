import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import incentive as it

# 데이터 로드
actual_weather = pd.read_csv('weather_actual.csv')
predictions = pd.read_csv('pred.csv')

# 예측 데이터에서 'model_id'와 'round' 열을 제외하고 실측 데이터와 병합
predictions = predictions.drop(['model_id', 'round'], axis=1)
combined_data = pd.merge(actual_weather, predictions, on='time')

# 특성 및 타겟 분리
X = combined_data.drop(['amount', 'time'], axis=1)  # 'time' 열과 타겟 변수 제외
y = combined_data['amount']  # 타겟 변수

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 초기화 및 훈련
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 테스트 데이터로 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'The Mean Squared Error on test set: {mse}')
mae = mean_absolute_error(y_test, y_pred)
print(f'The Mean Absolute Error on test set: {mae}')

# API로부터 예보 데이터 가져오기
api_data = it._get_weathers_forecasts10()  # API 함수를 호출하여 데이터를 가져옵니다.

# 'time' 열을 제거하고 예측 수행
api_data = api_data.drop('time', axis=1, errors='ignore')  # 'errors='ignore''를 추가하여 'time' 열이 없는 경우에도 오류가 나지 않도록 처리
predictions = model.predict(api_data)  # 예측 수행

# 예측 결과를 시각화
plt.figure(figsize=(10, 6))
plt.plot(predictions, label='Predictions')  # 예측값 플롯
plt.title('Generation Predictions')
plt.xlabel('Index')
plt.ylabel('Predicted Amount')
plt.legend()
plt.grid()
plt.show()
