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

def _10predictor_test():
    ##gen은 실제 발전량
    ##pred는 예측발전량
    ## train을 조정해서 최적의 오차율 만들기
    ## actual
    ## train x -> train y(학습) 그럼 test x 넣으면 test y는?(예측)
    train_x = pd.read_csv('weather_actual.csv', parse_dates=True)  # 학습시킬것 날씨량 (실제 날씨)
    train_y = pd.read_csv('gens.csv', parse_dates=True)  # 학습시킬것 발전량 (실제 발전량)
    test_x = it._get_weathers_forecasts10()  ##api 내일 일기예보 가져옴
    test_y = pd.read_csv('gens.csv', parse_dates=True)  ##api 로 내일


    #############################데이터 전처리##################################
    train_x = train_x[train_x.columns[1:]]  # 날씨실측정보
    train_y = train_y[train_y.columns[1:]]  # 발전실측정보

    ##########################
    test_x = test_x[test_x.columns[1:]]  # 1에 대한 일기예보만 살리기위해 일단 없앰

    #####################발전예측량에 대해서 어느모델에 대해 하는지###################################
    # 모델 예측량에따라서 산술 평균을 낼지, 가중치를 둬서 낼지, 아니면 하나만 선택해서 할지 고민 ㄱㄱ
    test_y1 = test_y[test_y.columns[1:2]]
    test_y2 = test_y[test_y.columns[2:3]]
    test_y3 = test_y[test_y.columns[3:4]]
    test_y4 = test_y[test_y.columns[4:5]]
    test_y5 = test_y[test_y.columns[5:]]

    test_ysum=pd.concat([test_y1,test_y2,test_y3,test_y4,test_y5],axis=1)
    test_y['Average']=test_ysum.mean(axis=1)
    test_y=test_y[test_y.columns[6:]]

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

    yhat = model.predict(test_x_reshaped)      ##test x(일기예보), yhat: 일기예보를 통해 예측된 발전량 -> gen_forecast

    yhat_original = scaler_y.inverse_transform(yhat)
    test_y_original = scaler_y.inverse_transform(test_y_scaled) ## test y 모델들의 발전량

    ## get_bids : 모델들의 인센티브값

    plt.figure(figsize=(15, 6))
    plt.plot(test_y_original, label='Actual') #실제 발전량
    plt.plot(yhat_original, label='Predicted')#예측된 발전량
    A=list(range(24))
    plt.xticks((A))
    plt.grid
    plt.legend()
    plt.title("10")
    plt.show()

    # LSTM의 MAE 계산
    mae_lstm = it.calculate_mae(test_y_original, yhat_original) #actual,predict
    print("MAE:",mae_lstm)

    # 예측량에대한 리스트
    pred = yhat_original
    pred = pred.tolist()
    pred = [item for sublist in pred for item in sublist]
    # 사이트에서 올려준 post
    it._post_bids(pred)

    print('a')

_10predictor_test()
