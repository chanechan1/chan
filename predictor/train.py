import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import requests
import json
import param as pa
import insentive as it
from datetime import datetime
import pytz

#실제기상-실제발전 / 기상예보-실제발전
def _10predictor():
    ##gen은 실제 발전량
    ##pred는 예측발전량
    train_x = pd.read_csv('weather_actual.csv', parse_dates=True)  # 학습시킬것 날씨량
    train_y = pd.read_csv('gens.csv', parse_dates=True)  # 학습시킬것 발전량
    test_x =  pd.read_csv('weather_forecast.csv', parse_dates=True)
    test_y =  pd.read_csv('gens.csv', parse_dates=True)

    #####################json to pandas(dataframe)#####################
    test_x = pd.DataFrame(test_x)
    test_y = pd.DataFrame(test_y)

    #############################데이터 전처리##################################
    train_x = train_x[train_x.columns[1:]]  # 날씨실측정보
    train_y = train_y[train_y.columns[1:]]  # 발전실측정보

    test_x = test_x[test_x.columns[2:]]  # 1에 대한 일기예보만 살리기위해 일단 없앰
    test_x = test_x.iloc[:11616]
    test_y=test_y[test_y.columns[1:]]

    #test_x['cloud'] = np.sqrt(test_x['cloud'])
    

    ##########################################################################
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

    yhat_original = scaler_y.inverse_transform(yhat)            #기상예측으로
    test_y_original = scaler_y.inverse_transform(test_y_scaled) #실제 발전량

    plt.figure(figsize=(15, 6))
    plt.plot(test_y_original, label='Actual')
    plt.plot(yhat_original, label='Predicted')
    plt.legend()
    plt.title("10")
    A = list(range(24))
    plt.xticks((A))
    plt.grid

    plt.show()

    # 예측량에대한 리스트
    pred = yhat_original
    pred=pred.tolist()
    pred = [item for sublist in pred for item in sublist]
    # 사이트에서 올려준 post
    it._post_bids(pred)


    # LSTM의 MAE 계산
    mae_lstm = it.calculate_mae(test_y_original, yhat_original)  # actual,predict

    print("MAE:", mae_lstm)

_10predictor()
