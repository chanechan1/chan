import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 실제 기상과 기상예보의 오차계산
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
    # inf면 제외
    error_df = error_df.replace([np.inf, -np.inf], np.nan)

    # 각 변수에 대한 평균 오차 계산 (NaN 값을 무시하고 평균 계산)
    average_error_per_variable = error_df.mean(axis=0, skipna=True)

    # 조건에 따라 예측 데이터 수정
    adjusted_forecast = test_x.copy()
    # for column in test_x.columns:
    #     adjusted_forecast[column] = np.where(test_x[column] > 80, test_x[column], test_x[column] / average_error_per_variable[column])

    # 첫 번째 열에 대해서만 그래프 그리기
    first_column = adjusted_forecast.columns[0]
    plt.figure(figsize=(12, 8))
    plt.plot(adjusted_forecast.index, adjusted_forecast[first_column], label=f'Adjusted Forecast - {first_column}')
    plt.plot(test_y.index, test_y[first_column], label=f'Actual Data - {first_column}', alpha=0.7)

    plt.title(f'Adjusted Weather Forecast vs Actual Data for {first_column}')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

_forecast_train()
