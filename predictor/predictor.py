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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pytz

##이전자료는 train이고 실제 할거는 test임 api를 받아와서 돌려볼 거는 test에 해당, x(다양한 변수들과)는 주어지는 날씨량이고 y(실제 발전량)는 결과임
##데이터 프레임으로 변환


########인센티브 가장 많은 정답지
def select_max_incentive_rows(group_df):
    return group_df.loc[group_df['incentive'].idxmax()]


def errorpredictor():##########인센티브와 오차를 구한 것에 대한 모델섞기
    train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)
    train_x = train_x.iloc[:11568]
    train_x = train_x[train_x.columns[2:]]  # 시간떼기

    ###여기서
    a = pd.read_csv('pred.csv', parse_dates=True)#
    a = a.iloc[:57840]
    a = a[a.columns[1:]]
    predictions = a.pivot(index='time', columns='model_id', values='amount')
    actual_values = pd.read_csv('gens.csv',parse_dates=True)##
    actual_values = actual_values[actual_values.columns[1:]]
    errors = predictions.to_numpy() - actual_values.to_numpy().reshape(-1, 1)
    errors=pd.DataFrame(errors)

    train_y = min_error_df
    ################################train set###########################


    test_x = it._get_weathers_forecasts10('2023-11-02')  ##api 로 내일 데이터 따오기
    test_x = test_x[test_x.columns[1:]]  ##시간 땐거

    model = Sequential()
    model.add(Dense(64, input_dim=13, activation='relu'))  # 입력층
    model.add(Dense(64, activation='relu'))  # 은닉층
    model.add(Dense(5, activation='sigmoid'))  # 출력층
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train_x, train_y = train_test_split(train_x, train_y)#,test_size=0.2)
    history = model.fit(train_x, train_y, epochs=150, batch_size=32, shuffle=1)  # , validation_data=(X_val, y_val))
    predictions = model.predict(test_x)
    predictions = pd.DataFrame(predictions)  ##24*5의 데이터프레임으로 바꾸고

    a = it._get_gen_forecasts10('2023-11-02')  ##api 로 내일 모델 입찰량
    a = pd.DataFrame(a)
    print(a)
    a = a[a.columns[1:]]
    a.columns = predictions.columns

    predamount = a * predictions  ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    pred1 = predamount.values.tolist()
    pred1 = [item[0] for item in pred1]
    print('a')

def _10predictor():
    ##gen은 실제 발전량  pred는 예측발전량 incentive는 각 모델들이 얼마나 잘 맞췄는지
    train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)  #실제 날씨로 풀려보고
    ###train_x = pd.read_csv('weather_actual.csv', parse_dates=True) #일기예보로도 풀려보고
    train_x = train_x.iloc[:11568]
    train_x = train_x[train_x.columns[2:]] #시간떼기

    #########train_y를 인센티브중 가장 높은 모델을 선택하게끔 데이터 전처리 하기
    train_y = pd.read_csv('incentive.csv', parse_dates=True)  #정답지 모델들의 인센티브값
    pivot_df = train_y.pivot(index='time', columns='model_id', values='incentive')

    # 인센티브가 가장 높은 모델에 대해서만 값을 1로 설정합니다.
    max_incentive = pivot_df.max(axis=1)
    filtered_data = pivot_df.apply(lambda x: (x == max_incentive).astype(int), axis=0)
    filtered_data_encoded = pd.get_dummies(filtered_data)
    df_normalized = filtered_data_encoded.div(filtered_data.sum(axis=1), axis=0)
    train_y = df_normalized

    ###################test_x 실제 생산량을 알 수 있는 날짜의 그것을 넣기############################
    test_x = it._get_weathers_forecasts10('2023-11-08')  ##api 로 내일 데이터 따오기
    test_x = test_x[test_x.columns[1:]] ##시간 땐거

    model = Sequential()
    model.add(Dense(64, input_dim=13, activation='relu'))  # 입력층
    model.add(Dense(64, activation='relu'))  # 은닉층
    model.add(Dense(5, activation='sigmoid'))  # 출력층

    X_val = train_x
    X_val = X_val.iloc[1680:]
    y_val = train_y
    y_val = y_val.iloc[1680:]

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train_x, train_y = train_test_split(train_x, train_y)#,test_size=0.2)
    history = model.fit(train_x, train_y, epochs=150, batch_size=32, shuffle=1)  # , validation_data=(X_val, y_val))
    predictions = model.predict(test_x)

    predictions = pd.DataFrame(predictions)  ##24*5의 데이터프레임으로 바꾸고
    predictions = predictions.div(predictions.sum(axis=1), axis=0)

    a = it._get_gen_forecasts10('2023-11-08')  ##api 로 내일 모델 입찰량
    a = pd.DataFrame(a)
    print(a)
    a = a[a.columns[1:]]
    a.columns = predictions.columns

    predamount = a * predictions  ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    pred1 = predamount.values.tolist()
    pred1 = [item[0] for item in pred1]


    # ####################################################17시 에대한 거################################
    #
    # ##gen은 실제 발전량  pred는 예측발전량 incentive는 각 모델들이 얼마나 잘 맞췄는지
    # train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)  # 실제 날씨로 풀려보고
    # ###train_x = pd.read_csv('weather_actual.csv', parse_dates=True) #일기예보로도 풀려보고
    # train_x = train_x.iloc[11568:]
    # train_x = train_x[train_x.columns[2:]]  # 시간떼기
    #
    # #########train_y를 인센티브중 가장 높은 모델을 선택하게끔 데이터 전처리 하기
    # train_y = pd.read_csv('incentive.csv', parse_dates=True)  # 정답지 모델들의 인센티브값
    # pivot_df = train_y.pivot(index='time', columns='model_id', values='incentive')
    #
    # # 인센티브가 가장 높은 모델에 대해서만 값을 1로 설정합니다.
    # max_incentive = pivot_df.max(axis=1)
    # filtered_data = pivot_df.apply(lambda x: (x == max_incentive).astype(int), axis=0)
    # filtered_data_encoded = pd.get_dummies(filtered_data)
    # df_normalized = filtered_data_encoded.div(filtered_data.sum(axis=1), axis=0)
    # train_y = df_normalized

    # ###################test_x 실제 생산량을 알 수 있는 날짜의 그것을 넣기############################
    # test_x = it._get_weathers_forecasts17('2023-11-05')  ##api 로 내일 데이터 따오기
    # test_x = test_x[test_x.columns[1:]]  ##시간 땐거
    #
    #
    #
    # model = Sequential()
    # model.add(Dense(64, input_dim=13, activation='relu'))  # 입력층
    # model.add(Dense(64, activation='relu'))  # 은닉층
    # model.add(Dense(5, activation='sigmoid'))  # 출력층
    #
    # X_val = train_x
    # X_val = X_val.iloc[1680:]
    # y_val = train_y
    # y_val = y_val.iloc[1680:]
    #
    #
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # #train_x, train_y = train_test_split(train_x, train_y)#,test_size=0.2)
    # history = model.fit(train_x, train_y, epochs=150, batch_size=32,shuffle=1)#, validation_data=(X_val, y_val))
    # predictions = model.predict(test_x)
    #
    # predictions = pd.DataFrame(predictions)  ##24*5의 데이터프레임으로 바꾸고
    # predictions = predictions.div(predictions.sum(axis=1), axis=0)
    #
    # a = it._get_gen_forecasts17('2023-11-05')  ##api 로 내일 모델 입찰량
    # a = pd.DataFrame(a)
    # print(a)
    # a = a[a.columns[1:]]
    # a.columns = predictions.columns
    #
    # predamount = a * predictions  ###가중치를 곱한 모델들을 섞은 예측발전량
    # predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    # pred2 = predamount.values.tolist()
    # pred2 = [item[0] for item in pred2]
    #
    #
    # x = list(range(len(pred2)))
    #
    # plt.figure(figsize=(6, 10))
    # # 그래프에 데이터 포인트와 선을 그립니다.
    # plt.plot(x, pred1,pred2, marker='o')
    #
    # # 각 데이터 포인트에 값을 표시합니다.
    # for i, value in enumerate(pred1):
    #     plt.text(x[i], value, str(value), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    # for i, value in enumerate(pred2):
    #     plt.text(x[i], value, str(value), fontsize=6, verticalalignment='bottom', horizontalalignment='right')
    # # 그래프에 제목과 축 라벨을 추가합니다.
    # plt.title('Predictions Plot')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # # 그리드를 표시합니다.
    # plt.grid(True)
    #
    # # 그래프를 화면에 보여줍니다.
    # plt.show()

    it._post_bids(pred1)
    print(pred1)
    #print(pred2)
    print('a')

#errorpredictor()
_10predictor()
