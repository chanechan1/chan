import pandas as pd
import numpy as np

    #실제 기상과 가중치 반영한 기상예보와의 오차 계산
def _forecast_train():
    # 데이터 로드
    test_x = pd.read_csv('weather_forecast.csv', parse_dates=True)
    test_y = pd.read_csv('weather_actual.csv', parse_dates=True)

    test_x = pd.DataFrame(test_x)
    test_y = pd.DataFrame(test_y)
    test_x = test_x[test_x.columns[2:]]
    test_y = test_y[test_y.columns[1:]]
    test_x = test_x.iloc[:11616]

    # 오차 계산 (절대값 사용하지 않음)
    error_df = (test_x - test_y) / test_y
    error_df = error_df.replace([np.inf, -np.inf], np.nan)
    average_error_per_variable = error_df.mean(axis=0, skipna=True)

    # 조건에 따라 예측 데이터 수정
    adjusted_forecast = test_x.copy()
    for column in test_x.columns:
        adjusted_forecast[column] = np.where(
            (test_x[column] > 60) | (test_x[column] < 20),
            test_x[column],
            test_x[column] / (1 + average_error_per_variable[column])
        )
    original_mae = np.mean(np.abs(test_x - test_y))

    # 조정된 예측 데이터와 실제 데이터 간의 MAE 계산
    adjusted_mae = np.mean(np.abs(adjusted_forecast - test_y))

    return original_mae, adjusted_mae

original_mae, adjusted_mae = _forecast_train()
print(f"Original Forecast MAE: {original_mae}")
print(f"Adjusted Forecast MAE: {adjusted_mae}")

# 더 나은 예측 선택
better_forecast = "Adjusted" if adjusted_mae < original_mae else "Original"
print(f"The better forecast is: {better_forecast}")
