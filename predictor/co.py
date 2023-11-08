import insentive as it
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def get_daily_amounts(date):
    # 날짜 필터링을 위한 문자열 포맷 조정
    data=pd.read_csv('gens_by_sugi.csv',parse_dates=True)
    start_time = f"{date} 01:00:00+09:00"
    end_time = f"{date} 24:00:00+09:00"
    # 주어진 날짜와 시간 범위에 해당하는 데이터 필터링
    daily_data = data[(data['time'] >= start_time) & (data['time'] < end_time)]
    # amount 값만 추출하여 리스트로 변환
    amounts = daily_data['amount'].tolist()
    amounts.append(0)
    return amounts

def error_predictor10(date,loss):
    a = pd.read_csv('pred.csv', parse_dates=True)  #
    a = a.iloc[:57840]
    a = a[a.columns[1:]]
    predictions = a.pivot(index='time', columns='model_id', values='amount')
    actual_values = pd.read_csv('gens.csv', parse_dates=True)  ##
    actual_values = actual_values[actual_values.columns[1:]]
    errors = predictions.to_numpy() -actual_values.to_numpy().reshape(-1, 1)
    errors = pd.DataFrame(errors)

    train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)
    train_x = train_x.iloc[:11568]
    train_x = train_x[train_x.columns[2:]]  # 시간떼기
    train_y = errors

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)

    # 모델 정의
    model = Sequential([
        Dense(256, activation='tanh', input_shape=(train_x.shape[1],)),  # 첫 번째 은닉층
        Dense(320, activation='sigmoid'),  # 두 번째 은닉층
        Dense(5, activation='linear')  # 출력층, 여기서는 5개의 출력을 가정합니다
    ])
    optimizer = Adam(learning_rate=0.0001)
    # 모델 컴파일
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(
        train_x, train_y,
        epochs=300,  # 큰 값으로 설정
        batch_size=32,
        verbose=1,
        validation_data=(val_x, val_y),
        callbacks=[early_stopping]  # 조기 종료 콜백 사용
    )
    test_x = it._get_weathers_forecasts10(date)  ##api 로 내일 데이터 따오기
    test_x = test_x[test_x.columns[1:]]  ##시간 땐거
    error_data = model.predict(test_x)  ## 기상정보를 넣었을때에 발생되는 오차

    ##########나온 오차값들로 최적의 가중치를 조정하는 부분################
    # 결과를 저장할 리스트
    optimal_weights_list = []

    # 목적 함수
    def objective(weights, errors):
        return np.abs(np.dot(weights, errors))

    # 제약 조건 함수들
    def constraint1(weights, errors):
        return 0.5 - np.dot(weights, errors)

    def constraint2(weights, errors):
        return np.dot(weights, errors) + 0.5

    def constraint3(weights):
        return 1.25 - sum(weights)

    # 각 행에 대해 최적화를 수행하는 루프
    for errors in error_data:
        # 초기 가중치
        initial_weights = np.full(len(errors), 1 / len(errors))

        # bounds를 각 행의 오차 수에 맞게 설정
        bounds = [(0.12, 1) for _ in range(len(errors))]

        # 제약 조건
        cons = [{'type': 'ineq', 'fun': constraint1, 'args': (errors,)},
                {'type': 'ineq', 'fun': constraint2, 'args': (errors,)},
                {'type': 'eq', 'fun': constraint3}]

        # 최적화 실행
        result = minimize(objective, initial_weights, args=(errors,), method='SLSQP', bounds=bounds, constraints=cons)

        # 결과 저장
        optimal_weights_list.append(result.x)

    optimal_weights_array = np.array(optimal_weights_list)
    optimal_weights_array = pd.DataFrame(optimal_weights_array)

    ###################구한 최적의 가중치랑 구하는 것##################
    a = it._get_gen_forecasts10(date)  ##api 로 내일 모델 입찰량
    a = pd.DataFrame(a)
    print(a)
    a = a[a.columns[1:]]
    a.columns = optimal_weights_array.columns

    predamount = a * optimal_weights_array  ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    pred = predamount.values.tolist()
    pred = [item[0] for item in pred]
    return pred
def error_predictor17(date,loss):
    a = pd.read_csv('pred.csv', parse_dates=True)  #
    a = a.iloc[57840:]
    a = a[a.columns[1:]]
    predictions = a.pivot(index='time', columns='model_id', values='amount')
    actual_values = pd.read_csv('gens.csv', parse_dates=True)  ##
    actual_values = actual_values[actual_values.columns[1:]]
    errors = predictions.to_numpy() -actual_values.to_numpy().reshape(-1, 1)
    errors = pd.DataFrame(errors)

    train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)
    train_x = train_x.iloc[11568:]
    train_x = train_x[train_x.columns[2:]]  # 시간떼기
    train_y = errors

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)

    # 모델 정의
    model = Sequential([
        Dense(256, activation='tanh', input_shape=(train_x.shape[1],)),  # 첫 번째 은닉층
        Dense(320, activation='sigmoid'),  # 두 번째 은닉층
        Dense(5, activation='linear')  # 출력층, 여기서는 5개의 출력을 가정합니다
    ])
    optimizer = Adam(learning_rate=0.0001)
    # 모델 컴파일
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(
        train_x, train_y,
        epochs=300,  # 큰 값으로 설정
        batch_size=32,  # 일반적으로 사용하는 초기 값
        verbose=1,
        validation_data=(val_x, val_y),
        callbacks=[early_stopping]  # 조기 종료 콜백 사용
    )

    test_x = it._get_weathers_forecasts17(date)  ##api 로 내일 데이터 따오기
    test_x = test_x[test_x.columns[1:]]  ##시간 땐거
    error_data = model.predict(test_x)  ## 기상정보를 넣었을때에 발생되는 오차

    ##########나온 오차값들로 최적의 가중치를 조정하는 부분################
    # 결과를 저장할 리스트
    optimal_weights_list = []

    # 목적 함수
    def objective(weights, errors):
        return np.abs(np.dot(weights, errors))

    # 제약 조건 함수들
    def constraint1(weights, errors):
        return 0.5 - np.dot(weights, errors)

    def constraint2(weights, errors):
        return np.dot(weights, errors) + 0.5

    def constraint3(weights):
        return 1.25 - sum(weights)

    # 각 행에 대해 최적화를 수행하는 루프
    for errors in error_data:
        # 초기 가중치
        initial_weights = np.full(len(errors), 1 / len(errors))

        # bounds를 각 행의 오차 수에 맞게 설정
        bounds = [(0.12, 1) for _ in range(len(errors))]

        # 제약 조건
        cons = [{'type': 'ineq', 'fun': constraint1, 'args': (errors,)},
                {'type': 'ineq', 'fun': constraint2, 'args': (errors,)},
                {'type': 'eq', 'fun': constraint3}]

        # 최적화 실행
        result = minimize(objective, initial_weights, args=(errors,), method='SLSQP', bounds=bounds, constraints=cons)

        # 결과 저장
        optimal_weights_list.append(result.x)

    optimal_weights_array = np.array(optimal_weights_list)
    optimal_weights_array = pd.DataFrame(optimal_weights_array)
    ###################구한 최적의 가중치랑 구하는 것##################
    a = it._get_gen_forecasts17(date)  ##api 로 내일 모델 입찰량
    a = pd.DataFrame(a)
    print(a)
    a = a[a.columns[1:]]
    a.columns = optimal_weights_array.columns

    predamount = a * optimal_weights_array  ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    ####가중치가 1이라서 그렇지
    pred = predamount.values.tolist()
    pred = [item[0] for item in pred]
    return pred
a=get_daily_amounts('2023-11-03')
#############그림그려서 알아보기###############
pred1=error_predictor10('2023-11-08','huber_loss')
pred2=error_predictor17('2023-11-08','huber_loss')
predamount =[(x + y) / 2 for x, y in zip(pred1, pred2)]
predamount = [round(num, 2) for num in predamount]
#error = [(a - b) for a, b in zip(get_daily_amounts('2023-11-03'), predamount)]
#print(error)
x = list(range(24))
plt.figure(figsize=(10, 6))
#그래프에 데이터 포인트와 선을 그립니다.
plt.plot(x, pred1 ,marker='o',color='orange')
plt.plot(x,pred2 , marker='o',color='red')
plt.plot(x,predamount,marker='o')
#plt.plot(x,error,  marker ='o',color = 'black')
for i, value in enumerate(predamount):
    plt.text(x[i], value, str(value), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
plt.axhline(y=6, color='blue', linestyle='-', label='y=6')
plt.axhline(y=8, color='g', linestyle='-', label='y=8')
plt.axhline(y=-6, color='blue', linestyle='-', label='y=-6')
plt.axhline(y=-8, color='g', linestyle='-', label='y=-8')
plt.title('Predictions Plot')
plt.xlabel('Index')
plt.ylabel('Value')
# 그리드를 표시합니다.
plt.grid(True)
it._post_bids(pred1)

# 그래프를 화면에 보여줍니다.
plt.show()



