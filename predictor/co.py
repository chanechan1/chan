import insentive as it
import pandas as pd
import numpy as np
import load as ld
from scipy.optimize import minimize
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model



def get_daily_amounts(date):
    data=pd.read_csv('gens_by_sugi.csv',parse_dates=True)
    start_time = f"{date} 01:00:00+09:00"
    end_time = f"{date} 24:00:00+09:00"
    daily_data = data[(data['time'] >= start_time) & (data['time'] < end_time)]
    # amount 값만 추출하여 리스트로 변환
    amounts = daily_data['amount'].tolist()
    amounts.append(0)
    return amounts

def error_predictor10(c,d):
    # test_x = it._get_weathers_forecasts10(date)  ##api 로 내일 데이터 따오기
    # test_x = test_x[test_x.columns[1:]]  ##시간 땐거
    ##error_data = model.predict(test_x)  ## 기상정보를 넣었을때에 발생되는 오차
    test_x=pd.read_csv('morning_forecast.csv',parse_dates=True)
    test_x=test_x[test_x.columns[1:14]]
    error_data=ld.load_and_predict_model10(test_x)


    ##########나온 오차값들로 최적의 가중치를 조정하는 부분################
    # 결과를 저장할 리스트

    optimal_weights_list = []

    # 목적 함수
    def objective(weights, errors):
        return np.abs(np.dot(weights, errors))

    # 제약 조건 함수들
    def constraint1(weights, errors):
        return 25 - np.dot(weights, errors)

    def constraint2(weights, errors):
        return np.dot(weights, errors) +25

    def constraint3(weights):
        return 1.1- sum(weights)

    # 각 행에 대해 최적화를 수행하는 루프
    for errors in error_data:
        # 초기 가중치
        initial_weights = np.full(len(errors), 1 / len(errors))

        # bounds를 각 행의 오차 수에 맞게 설정
        bounds = [(0.06, 1.2) for _ in range(len(errors))]

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
    # a = it._get_gen_forecasts10(date)  ##api 로 내일 모델 입찰량
    # a = pd.DataFrame(a)
    a=pd.read_csv('morning_forecast.csv',parse_dates=True)
    a=a[a.columns[14:]]

    print(a)
    #a = a[a.columns[1:]]
    a.columns = optimal_weights_array.columns

    predamount = a * optimal_weights_array  ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    pred = predamount.values.tolist()
    pred = [item[0] for item in pred]
    return pred
def error_predictor17(c,d):
    # test_x = it._get_weathers_forecasts17(date)  ##api 로 내일 데이터 따오기
    # test_x = test_x[test_x.columns[1:]]  ##시간 땐거
    # error_data = model.predict(test_x)  ## 기상정보를 넣었을때에 발생되는 오차

    test_x = pd.read_csv('evening_forecast.csv', parse_dates=True)
    test_x = test_x[test_x.columns[1:14]]
    error_data = ld.load_and_predict_model10(test_x)

    ##########나온 오차값들로 최적의 가중치를 조정하는 부분################
    # 결과를 저장할 리스트

    optimal_weights_list = []

    # 목적 함수
    def objective(weights, errors):
        return np.abs(np.dot(weights, errors))

    # 제약 조건 함수들
    def constraint1(weights, errors):
        return 25- np.dot(weights, errors)

    def constraint2(weights, errors):
        return np.dot(weights, errors) + 25

    def constraint3(weights):
        return 1.11- sum(weights)

    # 각 행에 대해 최적화를 수행하는 루프
    for errors in error_data:
        # 초기 가중치
        initial_weights = np.full(len(errors), 1 / len(errors))

        # bounds를 각 행의 오차 수에 맞게 설정
        bounds = [(0.02, 1.2) for _ in range(len(errors))]

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
    # a = it._get_gen_forecasts10(date)  ##api 로 내일 모델 입찰량
    # a = pd.DataFrame(a)
    a = pd.read_csv('evening_forecast.csv', parse_dates=True)
    a = a[a.columns[14:]]

    print(a)
    #a = a[a.columns[1:]]
    a.columns = optimal_weights_array.columns

    predamount = a * optimal_weights_array  ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    pred = predamount.values.tolist()
    pred = [item[0] for item in pred]
    return pred
def run(c,d):
    q = pd.read_csv('gens_by_sugi.csv', parse_dates=True)
    q = q[q.columns[1:]]
    q = q.values.tolist()
    q = [item[0] for item in q]

    pred1 = error_predictor10(c,d)
    pred2 = error_predictor17(c,d)
    predamount = [(x + y) / 2 for x, y in zip(pred1, pred2)]
    pred1 = [round(num, 1) for num in pred1]
    pred2 = [round(num, 1) for num in pred2]
    predamount = [round(num, 1) for num in predamount]
    a = it.calc_profit(q, pred1)
    b = it.calc_profit(q, pred2)
    d = it.calc_profit(q, q)
    e = it.calc_profit(q, predamount)
    x = list(range(len(q)))
    plt.figure(figsize=(22, 14))
    for x_value in x:
        plt.axvline(x=x_value, color='lightgray', linestyle='--', linewidth=0.5)
    plt.plot(x, q, marker='o', color='black')
    #plt.plot(x, pred1, marker='o', color='green')
    #plt.plot(x, pred2, marker='o', color='red')
    plt.plot(x,predamount,marker='o')

    for i, value in enumerate(predamount):
        plt.text(x[i], value, str(value), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, value in enumerate(q):
        plt.text(x[i], value, str(value), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    plt.title('Predictions Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # 그리드를 표시합니다.
    plt.grid(True)

    # 그래프를 화면에 보여줍니다.
    plt.show()
    g=it.calculate_mae(q,pred1)
    f=it.calculate_mae(q,pred2)
    h = it.calculate_mae(q, predamount)

    print(d)
    print(e)
    a = sum(a)
    b = sum(b)
    d = sum(d)
    print(a)
    print(b)
    #
    e=sum(e)
    print(e)
    # print(d)
    print(g)
    print(f)
    print(h)
    print(d)
    return e

# range_a = range(11)  # 0 ~ 10
# range_b = range(20)  # 0 ~ 15
range_c = range(10)  # 0 ~ 10
range_d = range(15)  # 0 ~ 15
#
# # 최댓값과 그때의 인덱스를 저장할 변수
max_value = float('+inf')
max_indices = None
#
# # 모든 조합에 대해 run 함수 실행
# for a in range_a:
#     for b in range_b:
# for c in range_c:
#     for d in range_d:
#         result = run(c,d)
#         if result < max_value:
#             max_value = result
#             max_indices = (c,d)
# print(max_value)
# print(max_indices)



run(1,1)
