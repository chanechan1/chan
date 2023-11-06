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
import functions as func
from datetime import datetime
import pytz

##이전자료는 train이고 실제 할거는 test임 api를 받아와서 돌려볼 거는 test에 해당, x(다양한 변수들과)는 주어지는 날씨량이고 y(실제 발전량)는 결과임
##데이터 프레임으로 변환


########인센티브 가장 많은 정답지
def select_max_incentive_rows(group_df):
    return group_df.loc[group_df['incentive'].idxmax()]
def _10predictor():
    ##gen은 실제 발전량  pred는 예측발전량 incentive는 각 모델들이 얼마나 잘 맞췄는지
    train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)  #실제 날씨로 풀려보고
    ###train_x = pd.read_csv('weather_actual.csv', parse_dates=True) #일기예보로도 풀려보고
    train_x = train_x.iloc[:11568]
    train_x = train_x[train_x.columns[2:]] #시간떼기

    #########train_y를 인센티브중 가장 높은 모델을 선택하게끔 데이터 전처리 하기
    train_y = pd.read_csv('incentive.csv', parse_dates=True)  #정답지 모델들의 인센티브값
    train_y = train_y.groupby(train_y.index // 5).apply(select_max_incentive_rows)##가장 인센티브 많은
    train_y=train_y['model_id'].to_frame()
    ##5개의 정수만 나타나게 하기위함
    train_y = to_categorical(train_y, 5)
    train_y = pd.DataFrame(train_y)

    ###################test_x 실제 생산량을 알 수 있는 날짜의 그것을 넣기############################
    test_x = func._get_weathers_forecasts10('2023-11-02')  ##api 로 내일 데이터 따오기
    test_x = test_x[test_x.columns[1:]] ##시간 땐거

    ########################아래 상관도가 맞는지 확인하기 위해서 ###########################
    # test_x=pd.DataFrame(test_x)
    # cor10=pa.cor10
    # test_x=test_x.divide(cor10,axis=1)
    ######test_y는 각 모델을 선정할 것 같은날짜의 get_bidresult를 통해 가장 인센티브가 높은 값을 설정
    test_y=func._get_bids_result('2023-11-04') ##1102의 결과가 들어가 있는데 여기서 가장 큰 것만
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
    model.fit(train_x_reshaped, train_y, epochs=50, batch_size=64,shuffle=1)
    predictions = model.predict(test_x_reshaped) ##이게 확률에 대한


    predictions=pd.DataFrame(predictions) ##24*5의 데이터프레임으로 바꾸고
    a = func._get_gen_forecasts10('2023-11-02')  ##api 로 내일 모델 입찰량
    a= pd.DataFrame(a)
    a = a[a.columns[1:]]
    a.columns = predictions.columns
    predamount = a * predictions ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount=predamount.sum(axis=1).to_frame(name='predamounts')
    pred=predamount.values.tolist()
    pred = [item[0] for item in pred]
    #pred=pred.f
    x = list(range(len(pred)))

    # 그래프에 데이터 포인트와 선을 그립니다.
    plt.plot(x, pred, marker='o')

    # 각 데이터 포인트에 값을 표시합니다.
    for i, value in enumerate(pred):
        plt.text(x[i], value, str(value), fontsize=8, verticalalignment='bottom', horizontalalignment='right')

    # 그래프에 제목과 축 라벨을 추가합니다.
    plt.title('Predictions Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 그리드를 표시합니다.
    plt.grid(True)

    # 그래프를 화면에 보여줍니다.
    plt.show()

    #func._post_bids(pred)
    print(pred)
    print('a')
_10predictor()









