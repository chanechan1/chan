import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import requests
import json
import param as pa
import functions as func
from datetime import datetime
import pytz

def _10predictor():

    # 데이터 로드
    forecast_data = pd.read_csv('weather_forecast.csv', parse_dates=True)
    actual_data = pd.read_csv('weather_actual.csv', parse_dates=True)
    #test_x = func._get_weathers_forecasts10()  ##api 내일 일기예보 가져옴
    test_x = func._get_weathers_forecasts17()

    # 여기에서 데이터를 전처리하고, 필요한 경우 병합합니다.
    forecast_data = pd.DataFrame(forecast_data)
    actual_data = pd.DataFrame(actual_data)
    test_x = pd.DataFrame(test_x)
    forecast_data = forecast_data[forecast_data.columns[2:]]
    train_x = actual_data[actual_data.columns[1:]]
    #train_y = forecast_data.iloc[:11568]                  #round 1
    train_y = forecast_data.iloc[11568:]
    test_x = test_x.iloc[:, 1:]

    print('a')
    # Scaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # 학습 데이터 준비
    train_x_values = train_x.values
    train_x_scaled = scaler_x.fit_transform(train_x_values)

    train_y_values = train_y.values
    train_y_scaled = scaler_y.fit_transform(train_y_values)

    test_x_values = test_x.values
    test_x_scaled = scaler_x.transform(test_x_values)

    # LSTM 입력 형태에 맞게 데이터 변환
    train_x_reshaped = train_x_scaled.reshape((train_x_scaled.shape[0], 1, train_x_scaled.shape[1]))
    test_x_reshaped = test_x_scaled.reshape((test_x_scaled.shape[0], 1, test_x_scaled.shape[1]))

    input_shape = (1, train_x_scaled.shape[1])

    # LSTM 모델 구축
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    # 모델 훈련
    model.fit(train_x_reshaped, train_y_scaled, epochs=10, batch_size=32, verbose=2, shuffle=False)

    # 새로운 일기 예보에 대한 보정된 예보 생성
    yhat = model.predict(test_x_reshaped)

    yhat_expanded = np.tile(yhat, (1, 13))  # `yhat` 배열을 13개 컬럼으로 복제합니다.

    # 그 다음, 확장된 `yhat` 배열에 대해 `inverse_transform`을 호출합니다.
    yhat_original = scaler_y.inverse_transform(yhat_expanded)

    # 보정된 예보 리스트 출력
    corrected_forecast = yhat_original.flatten().tolist()  # 리스트 형태로 변환
    print('Corrected weather forecast:', corrected_forecast)

    print('a')
    return corrected_forecast  # 보정된 일기예보 리스트 반환

def _17predictor():

    train_x = pd.read_csv('weather_actual.csv', parse_dates=True)  # 학습시킬것 날씨량
    train_y = pd.read_csv('gens.csv', parse_dates=True)  # 학습시킬것 발전량
    test_x = func._get_weathers_forecasts17()  ##api 로 내일 데이터 따오기
    test_y = func._get_gen_forecasts17()  ##api 로 내일

    #####################json to pandas(dataframe)#####################
    test_x = pd.DataFrame(test_x)
    test_y = pd.DataFrame(test_y)
    ##test_x = pd.read_csv('weather_forecast.csv', index_col=1, parse_dates=True) api에서 내일 따올 기상예보
    ##test_y = pd.read_csv('pred.csv', parse_dates=True)

    #############################데이터 전처리##################################
    train_x = train_x[train_x.columns[1:]]  # 날씨실측정보
    train_y = train_y[train_y.columns[1:]]  # 발전실측정보

    ##########################
    test_x = test_x[test_x.columns[1:]]  # 1에 대한 일기예보만 살리기위해 일단 없앰

    #####################발전예측량에 대해서 어느모델에 대해 하는지###################################

    test_y1 = test_y[test_y.columns[1:2]]
    test_y2 = test_y[test_y.columns[2:3]]
    test_y3 = test_y[test_y.columns[3:4]]
    test_y4 = test_y[test_y.columns[4:5]]
    test_y5 = test_y[test_y.columns[5:]]
    test_ysum = pd.concat([test_y1, test_y2, test_y3, test_y4, test_y5], axis=1)
    test_y['Average'] = test_ysum.mean(axis=1)
    test_y = test_y[test_y.columns[6:]]
    # test_x=test_x.iloc[:11616]        #1에 대한 것만 살림
    # test_y=test_y.iloc[:58080]
    # test_y=test_y[test_y.columns[3:]] #amount만 남기고
    # test_y=test_y.iloc[2::5]

    # 정규화
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    ##값들만 추출 학습에 쓰이는 형태로 쓰기
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

    #
    input_shape = (1, train_x_scaled.shape[1])

    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    history = model.fit(train_x_reshaped, train_y_scaled, epochs=100, batch_size=32,
                        validation_data=(test_x_reshaped, test_y_scaled), verbose=2, shuffle=False)

    yhat = model.predict(test_x_reshaped)

    yhat_original = scaler_y.inverse_transform(yhat)
    test_y_original = scaler_y.inverse_transform(test_y_scaled)

    plt.figure(figsize=(15, 6))
    plt.plot(test_y_original, label='Actual')
    plt.plot(yhat_original, label='Predicted')
    plt.legend()
    plt.title("17")
    A = list(range(24))
    plt.xticks((A))
    plt.grid

    plt.show()

    # 예측량에대한 리스트
    pred = yhat_original
    pred=pred.tolist()
    pred = [item for sublist in pred for item in sublist]
    # 사이트에서 올려준 post
    func._post_bids(pred)
    # LSTM의 MAE 계산
    mae_lstm = func.calculate_mae(test_y_original, yhat_original)  # actual,predict

    print("MAE:", mae_lstm)


_10predictor()
#_17predictor()








