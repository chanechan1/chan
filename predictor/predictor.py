import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import requests
import json
import param as pa
import insentive as it
from datetime import datetime
import pytz

##이전자료는 train이고 실제 할거는 test임 api를 받아와서 돌려볼 거는 test에 해당, x(다양한 변수들과)는 주어지는 날씨량이고 y(실제 발전량)는 결과임
##데이터 프레임으로 변환


########인센티브 가장 많은 정답지
def select_max_incentive_rows(group_df):
    return group_df.loc[group_df['incentive'].idxmax()]
def _10predictor():
    ##gen은 실제 발전량  pred는 예측발전량 incentive는 각 모델들이 얼마나 잘 맞췄는지
    train_x = pd.read_csv('weather_actual.csv', parse_dates=True)  #실제 날씨로 풀려보고
    ###train_x = pd.read_csv('weather_actual.csv', parse_dates=True) #일기예보로도 풀려보고
    train_x = train_x[train_x.columns[1:]]  # 날씨실측정보

    #########train_y를 인센티브중 가장 높은 모델을 선택하게끔 데이터 전처리 하기
    train_y = pd.read_csv('incentive.csv', parse_dates=True)  #정답지 모델들의 인센티브값
    train_y = train_y.groupby(train_y.index // 5).apply(select_max_incentive_rows)##가장 인센티브 많은
    train_y=train_y['model_id'].to_frame()
    ##5개의 정수만 나타나게 하기위함
    train_y = to_categorical(train_y, 5)
    train_y = pd.DataFrame(train_y)
    ###################test_x 실제 생산량을 알 수 있는 날짜의 그것을 넣기############################
    test_x = it._get_weathers_forecasts10()  ##api 로 내일 데이터 따오기
    test_x = test_x[test_x.columns[1:]] ##시간 땐거

    ########################아래 상관도가 맞는지 확인하기 위해서 ###########################
    # test_x=pd.DataFrame(test_x)
    # cor10=pa.cor10
    # test_x=test_x.divide(cor10,axis=1)
    ######test_y는 각 모델을 선정할 것 같은날짜의 get_bidresult를 통해 가장 인센티브가 높은 값을 설정
    test_y=it._get_bids_result() ##1102의 결과가 들어가 있는데 여기서 가장 큰 것만
    df_model_selection = pd.DataFrame(0, index=test_y.index, columns=test_y.columns[1:])
    # 각 행에 대해 반복
    for index, row in test_y.iterrows():
        min_error = float('inf')
        min_model = None
        # 각 모델에 대해 반복
        for model in test_y.columns[1:]:  # 'time' 열을 제외한 모든 열
            error = row[model]['error']
            if error < min_error:
                min_error = error
                min_model = model
        # 해당 모델에 1을 부여
        df_model_selection.at[index, min_model] = 1.0

    # 결과 데이터 프레임에 시간을 추가합니다.
    df_model_selection.insert(0, 'time', test_y['time'])
    test_y=df_model_selection
    test_y = test_y[test_y.columns[1:]]
    test_y = test_y.astype('float32')

    scaler_x = MinMaxScaler()
    train_x_values = train_x.values
    train_x_scaled = scaler_x.fit_transform(train_x_values)
    test_x_values = test_x.values
    test_x_scaled = scaler_x.transform(test_x_values)
    train_x_reshaped = train_x_scaled.reshape((train_x_scaled.shape[0], 1, train_x_scaled.shape[1]))
    test_x_reshaped = test_x_scaled.reshape((test_x_scaled.shape[0], 1, test_x_scaled.shape[1]))

    train_x = train_x.values.reshape((train_x.shape[0], 1, train_x.shape[1]))

    test_x = test_x.values.reshape((test_x.shape[0], 1, test_x.shape[1]))

    input_shape = (1, train_x.shape[1])
    ##값들만 추출 학습에 쓰이는 형태로 쓰기
    model = Sequential()
    model.add(LSTM(50, input_shape=(1,13)))
    model.add(Dense(5, activation='softmax'))

    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 모델 학습
    model.fit(train_x_reshaped, train_y, epochs=300, batch_size=64)
    predictions = model.predict(test_x_reshaped)

    predicted_classes = predictions.argmax(axis=1)
    yhat=predicted_classes

    #####학습시킨걸로 인센티브 먹게끔 model의
    selected_values = []
    goodmodel = yhat ###학습을 통한 개굿모델
    goodmodel = goodmodel.tolist()
    a = it._get_gen_forecasts10()  ##api 로 내일 모델 입찰량
    a= pd.DataFrame(a)

    for i, model_index in enumerate(goodmodel):
        selected_value = a.iloc[i, model_index + 1]  # +1은 'time' 컬럼을 고려하여 인덱스를 조정
        selected_values.append(selected_value)


    a = pd.DataFrame(selected_values)



    #yhat_original[yhat_original<0]=0




    # 예측량에대한 리스트

    it._post_bids(selected_values)

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

    cor10 = pa.cor10
    test_x = test_x.divide(cor10, axis=1)

    #####################발전예측량에 대해서 어느모델에 대해 하는지###################################
    # 모델 예측량에따라서 산술 평균을 낼지, 가중치를 둬서 낼지, 아니면 하나만 선택해서 할지 고민 ㄱㄱ

    #################가장좋은 모델에 대해서만 test_y를 ######################

    goodmodel = [0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 3, 1, 3, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0]

    selected_values = []

    for i, model_index in enumerate(goodmodel):
        selected_value = test_y.iloc[i, model_index + 1]  # +1은 'time' 컬럼을 고려하여 인덱스를 조정
        selected_values.append(selected_value)

    test_y = pd.DataFrame(selected_values)


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

    history = model.fit(train_x_reshaped, train_y_scaled, epochs=300, batch_size=64,
                        validation_data=(test_x_reshaped, test_y_scaled), verbose=2, shuffle=False)

    yhat = model.predict(test_x_reshaped)

    yhat_original = scaler_y.inverse_transform(yhat)
    test_y_original = scaler_y.inverse_transform(test_y_scaled)
    yhat_original[yhat_original < 0] = 0



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
    ##it._post_bids(pred)


    # LSTM의 MAE 계산
    mae_lstm = it.calculate_mae(test_y_original, yhat_original)  # actual,predict

    print("MAE:", mae_lstm)


_10predictor()
#_17predictor()








