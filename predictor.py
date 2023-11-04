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

##이전자료는 train이고 실제 할거는 test임 api를 받아와서 돌려볼 거는 test에 해당, x(다양한 변수들과)는 주어지는 날씨량이고 y(실제 발전량)는 결과임
##데이터 프레임으로 변환


def _10predictor():
    ##gen은 실제 발전량
    ##pred는 예측발전량
    train_x = pd.read_csv('weather_actual.csv', parse_dates=True)  # 학습시킬것 날씨량
    train_y = pd.read_csv('gens.csv', parse_dates=True)  # 학습시킬것 발전량
    test_x = it._get_weathers_forecasts10()  ##api 로 내일 데이터 따오기
    test_y = it._get_gen_forecasts10()  ##api 로 내일

    #############################데이터 전처리##################################
    train_x = train_x[train_x.columns[1:]]  # 날씨실측정보
    train_y = train_y[train_y.columns[1:]]  # 발전실측정보

    ##########################
    test_x = test_x[test_x.columns[1:]]

    #################가장좋은 모델에 대해서만 test_y를 ######################

    goodmodel = [0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 3, 1, 3, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0]

    selected_values = []

    for i, model_index in enumerate(goodmodel):
        selected_value = test_y.iloc[i, model_index + 1]  # +1은 'time' 컬럼을 고려하여 인덱스를 조정
        selected_values.append(selected_value)

    test_y = pd.DataFrame(selected_values)

    # #####################발전예측량에 대해서 어느모델에 대해 하는지###################################
    # # 모델 예측량에따라서 산술 평균을 낼지, 가중치를 둬서 낼지, 아니면 하나만 선택해서 할지 고민 ㄱㄱ
    # test_y1 = test_y[test_y.columns[1:2]]
    # test_y2 = test_y[test_y.columns[2:3]]
    # test_y3 = test_y[test_y.columns[3:4]]
    # test_y4 = test_y[test_y.columns[4:5]]
    # test_y5 = test_y[test_y.columns[5:]]

    # test_ysum=pd.concat([test_y1,test_y2,test_y3,test_y4,test_y5],axis=1)
    # test_y['Average']=test_ysum.mean(axis=1)
    # test_y=test_y[test_y.columns[6:]]

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

    history = model.fit(train_x_reshaped, train_y_scaled, epochs=300, batch_size=32,
                        validation_data=(test_x_reshaped, test_y_scaled), verbose=2, shuffle=False)

    yhat = model.predict(test_x_reshaped)

    yhat_original = scaler_y.inverse_transform(yhat)
    test_y_original = scaler_y.inverse_transform(test_y_scaled)

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
def _17predictor():
    ##gen은 실제 발전량
    ##pred는 예측발전량
    train_x = pd.read_csv('weather_actual.csv', parse_dates=True)  # 학습시킬것 날씨량
    train_y = pd.read_csv('gens.csv', parse_dates=True)  # 학습시킬것 발전량
    test_x = it._get_weathers_forecasts17()  ##api 로 내일 데이터 따오기
    test_y = it._get_gen_forecasts17()  ##api 로 내일

    #####################json to pandas(dataframe)#####################
    test_x = pd.DataFrame(test_x)
    test_y = pd.DataFrame(test_y)

    #############################데이터 전처리##################################
    train_x = train_x[train_x.columns[1:]]  # 날씨실측정보
    train_y = train_y[train_y.columns[1:]]  # 발전실측정보

    ##########################
    test_x = test_x[test_x.columns[1:]]  # 1에 대한 일기예보만 살리기위해 일단 없앰

    #####################발전예측량에 대해서 어느모델에 대해 하는지###################################
    # 모델 예측량에따라서 산술 평균을 낼지, 가중치를 둬서 낼지, 아니면 하나만 선택해서 할지 고민 ㄱㄱ

    #################가장좋은 모델에 대해서만 test_y를 ######################

    goodmodel = [0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 3, 1, 3, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0]

    selected_values = []

    for i, model_index in enumerate(goodmodel):
        selected_value = test_y.iloc[i, model_index + 1]  # +1은 'time' 컬럼을 고려하여 인덱스를 조정
        selected_values.append(selected_value)

    test_y = pd.DataFrame(selected_values)
    ########산술평균##########
    # test_y1 = test_y[test_y.columns[1:2]]
    # test_y2 = test_y[test_y.columns[2:3]]
    # test_y3 = test_y[test_y.columns[3:4]]
    # test_y4 = test_y[test_y.columns[4:5]]
    # test_y5 = test_y[test_y.columns[5:]]
    # test_ysum = pd.concat([test_y1, test_y2, test_y3, test_y4, test_y5], axis=1)
    # test_y['Average'] = test_ysum.mean(axis=1)
    # test_y = test_y[test_y.columns[6:]]

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

    history = model.fit(train_x_reshaped, train_y_scaled, epochs=300, batch_size=32,
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
    it._post_bids(pred)


    # LSTM의 MAE 계산
    mae_lstm = it.calculate_mae(test_y_original, yhat_original)  # actual,predict

    print("MAE:", mae_lstm)


_10predictor()
_17predictor()







