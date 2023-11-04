# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import incentive as it
#
# # 예측 결과를 포함하는 CSV 파일 로드
# model_predictions = pd.read_csv('pred.csv')
#
# # 모델 예측값을 분할하기 위해 각 모델의 예측값을 담을 빈 리스트 생성
# model_predictions_list = [[] for _ in range(5)]  # 모델 1부터 5까지
#
# # CSV 파일의 각 행을 5개의 모델로 분할
# for _, row in model_predictions.iterrows():
#
#     split_row = np.array_split(row, 5)
#     for i in range(5):
#         # 각 모델 예측 리스트에 분할된 값을 추가
#         model_predictions_list[i].append(split_row[i].tolist())
#
# # 각 모델의 예측 결과를 데이터프레임으로 변환
# ensemble_df = pd.DataFrame({
#     'model1': model_predictions_list[0],
#     'model2': model_predictions_list[1],
#     'model3': model_predictions_list[2],
#     'model4': model_predictions_list[3],
#     'model5': model_predictions_list[4]
# })
#
# # 각 모델의 예측 결과는 현재 리스트 형태이므로 np.array로 변환하여 모델에 적합하게 만들어줌
# ensemble_df = ensemble_df.applymap(lambda cell: np.array(cell))
#
# # 실제 테스트 데이터를 CSV 파일에서 로드
# test_y = pd.read_csv('gens.csv')  # 'gen.csv'를 실제 파일 경로로 변경해야 합니다.
# test_y_numeric = test_y['amount'].values
#
# # 선형 회귀 모델을 사용하여 앙상블에 대한 가중치를 학습
# lr = LinearRegression()
#
# lr.fit(ensemble_df, test_y_numeric)
#
# # 가중치로 예측 결과를 결합하여 앙상블 예측 생성
# ensemble_predictions = lr.predict(ensemble_df)
#
# # 성능 평가
# mae = it.calculate_mae(test_y_numeric, ensemble_predictions.flatten())
#
# print("앙상블 모델의 MAE:", mae)
#
# # 각 모델에 대한 가중치 출력 (선택사항)
# print("각 모델의 가중치: ", lr.coef_)
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import requests
import json
import param as pa
import incentive as it
from datetime import datetime
import pytz

def evaluate_model(epochs, batch_size, units):
    train_x = pd.read_csv('weather_actual.csv', parse_dates=True)
    train_y = pd.read_csv('gens.csv', parse_dates=True)
    test_x = it._get_weathers_forecasts10()
    test_y = it._get_gen_forecasts10()

    train_x = train_x[train_x.columns[1:]]
    train_y = train_y[train_y.columns[1:]]

    test_x = test_x[test_x.columns[1:]]
    test_y1 = test_y[test_y.columns[1:2]]
    test_y2 = test_y[test_y.columns[2:3]]
    test_y3 = test_y[test_y.columns[3:4]]
    test_y4 = test_y[test_y.columns[4:5]]
    test_y5 = test_y[test_y.columns[5:]]

    test_ysum=pd.concat([test_y1,test_y2,test_y3,test_y4,test_y5],axis=1)
    test_y['Average']=test_ysum.mean(axis=1)
    test_y=test_y[test_y.columns[6:]]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

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

    input_shape = (1, train_x_scaled.shape[1])

    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    history = model.fit(train_x_reshaped, train_y_scaled, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_x_reshaped, test_y_scaled), verbose=2, shuffle=False)

    yhat = model.predict(test_x_reshaped)

    yhat_original = scaler_y.inverse_transform(yhat)
    test_y_original = scaler_y.inverse_transform(test_y_scaled)

    mae_lstm = it.calculate_mae(test_y_original, yhat_original)

    return mae_lstm

# 하이퍼파라미터 조합을 정의합니다.
epochs_list = [100, 200, 300]
batch_size_list = [32, 64]
units_list = [50, 100, 200]

results = []

for epochs in epochs_list:
    for batch_size in batch_size_list:
        for units in units_list:
            mae = evaluate_model(epochs, batch_size, units)
            results.append({
                'epochs': epochs,
                'batch_size': batch_size,
                'units': units,
                'mae': mae
            })

# 결과 출력
for result in results:
    print(f"Epochs: {result['epochs']}, Batch Size: {result['batch_size']}, Units: {result['units']}, MAE: {result['mae']}")
