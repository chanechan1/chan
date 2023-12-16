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
    test_x = pd.read_csv('morning_forecast.csv', parse_dates=True)
    test_x = test_x[test_x.columns[1:14]]
    error_data = ld.load_and_predict_model10(test_x)

    #24개가 아닐때
    #나온 24개의 데이터를 파트별로 나누는 파트
    part1, part2, part3 = [], [], []

    # Looping through each group of 24 rows and splitting them into parts
    for i in range(0, error_data.shape[0], 24):
        group = error_data[i:i + 24, :]
        part1.append(group[:10, :])  # First 10 rows of the group
        part2.append(group[10:15, :])  # Next 5 rows of the group
        part3.append(group[15:, :])  # Last 9 rows of the group

    # To create continuous arrays, we use numpy.vstack
    part1= np.vstack(part1)
    part2 = np.vstack(part2)
    part3 = np.vstack(part3)
    optimal_weights_list1 = []
    def objective1(weights, errors):
        return np.abs(np.dot(weights, errors))
    def constraint11(weights, errors):
        return 25- np.dot(weights, errors)
    def constraint21(weights, errors):
        return np.dot(weights, errors) + 25
    def constraint31(weights):
        return 1.14- sum(weights)
    for errors in part1:
        initial_weights = np.full(len(errors), 1 / len(errors))
        bounds = [(0.089, 1.5) for _ in range(len(errors))]
        cons = [{'type': 'ineq', 'fun': constraint11, 'args': (errors,)},
                {'type': 'ineq', 'fun': constraint21, 'args': (errors,)},
                {'type': 'eq', 'fun': constraint31}]
        result = minimize(objective1, initial_weights, args=(errors,), method='SLSQP', bounds=bounds, constraints=cons)
        optimal_weights_list1.append(result.x)
    optimal_weights_array1 = np.array(optimal_weights_list1)

    ##########part2#####################
    optimal_weights_list2 = []
    def objective2(weights, errors):
        return np.abs(np.dot(weights, errors))
    def constraint12(weights, errors):
        return 25 - np.dot(weights, errors)
    def constraint22(weights, errors):
        return np.dot(weights, errors) + 25
    def constraint32(weights):
        return 1.11 - sum(weights)
    for errors in part2:
        initial_weights = np.full(len(errors), 1 / len(errors))
        bounds = [(0.02, 1.5) for _ in range(len(errors))]
        cons = [{'type': 'ineq', 'fun': constraint12, 'args': (errors,)},
                {'type': 'ineq', 'fun': constraint22, 'args': (errors,)},
                {'type': 'eq', 'fun': constraint32}]
        result = minimize(objective2, initial_weights, args=(errors,), method='SLSQP', bounds=bounds, constraints=cons)
        optimal_weights_list2.append(result.x)
    optimal_weights_array2 = np.array(optimal_weights_list2)

    ##########part3#########################
    optimal_weights_list3 = []

    def objective3(weights, errors):
        return np.abs(np.dot(weights, errors))
    def constraint13(weights, errors):
        return 25 - np.dot(weights, errors)
    def constraint23(weights, errors):
        return np.dot(weights, errors) + 25
    def constraint33(weights):
        return 1.2 - sum(weights)

    for errors in part3:
        initial_weights = np.full(len(errors), 1 / len(errors))
        bounds = [(0.089, 1.5) for _ in range(len(errors))]
        cons = [{'type': 'ineq', 'fun': constraint13, 'args': (errors,)},
                {'type': 'ineq', 'fun': constraint23, 'args': (errors,)},
                {'type': 'eq', 'fun': constraint33}]
        result = minimize(objective3, initial_weights, args=(errors,), method='SLSQP', bounds=bounds, constraints=cons)
        optimal_weights_list3.append(result.x)
    optimal_weights_array3 = np.array(optimal_weights_list3)

    reconstructed_data = []

    for i in range(10):  # Assuming 10 groups as in the original splitting
        start_idx = i * 10
        end_idx = start_idx + 10
        reconstructed_data.append(optimal_weights_array1[start_idx:end_idx, :])

        start_idx = i * 5
        end_idx = start_idx + 5
        reconstructed_data.append(optimal_weights_array2[start_idx:end_idx, :])

        start_idx = i * 9
        end_idx = start_idx + 9
        reconstructed_data.append(optimal_weights_array3[start_idx:end_idx, :])

    # Combining all the parts
    optimal_weights_array = np.vstack(reconstructed_data)
    optimal_weights_array = pd.DataFrame(optimal_weights_array)

    ###################구한 최적의 가중치랑 구하는 것##################
    a=pd.read_csv('morning_forecast.csv',parse_dates=True)
    a = a[a.columns[14:]]
    a.columns = optimal_weights_array.columns
    a.index = optimal_weights_array.index

    predamount = a * optimal_weights_array  ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    pred = predamount.values.tolist()
    pred = [item[0] for item in pred]

    return pred
def error_predictor17(c,d):
    test_x = pd.read_csv('evening_forecast.csv', parse_dates=True)
    test_x = test_x[test_x.columns[1:14]]


    error_data = ld.load_and_predict_model10(test_x)

    # 나온 24개의 데이터를 파트별로 나누는 파트

    part1, part2, part3 = [], [], []

    # Looping through each group of 24 rows and splitting them into parts
    for i in range(0, error_data.shape[0], 24):
        group = error_data[i:i + 24, :]
        part1.append(group[:10, :])  # First 10 rows of the group
        part2.append(group[10:15, :])  # Next 5 rows of the group
        part3.append(group[15:, :])  # Last 9 rows of the group

    # To create continuous arrays, we use numpy.vstack
    part1= np.vstack(part1)
    part2 = np.vstack(part2)
    part3 = np.vstack(part3)
    optimal_weights_list1 = []

    def objective1(weights, errors):
        return np.abs(np.dot(weights, errors))

    def constraint11(weights, errors):
        return 25 - np.dot(weights, errors)

    def constraint21(weights, errors):
        return np.dot(weights, errors) + 25

    def constraint31(weights):
        return 1.1 - sum(weights)

    for errors in part1:
        initial_weights = np.full(len(errors), 1 / len(errors))
        bounds = [(0.15, 1.5) for _ in range(len(errors))]
        cons = [{'type': 'ineq', 'fun': constraint11, 'args': (errors,)},
                {'type': 'ineq', 'fun': constraint21, 'args': (errors,)},
                {'type': 'eq', 'fun': constraint31}]
        result = minimize(objective1, initial_weights, args=(errors,), method='SLSQP', bounds=bounds, constraints=cons)
        optimal_weights_list1.append(result.x)
    optimal_weights_array1 = np.array(optimal_weights_list1)

    ##########part2#####################
    optimal_weights_list2 = []

    def objective2(weights, errors):
        return np.abs(np.dot(weights, errors))

    def constraint12(weights, errors):
        return 25 - np.dot(weights, errors)

    def constraint22(weights, errors):
        return np.dot(weights, errors) + 25

    def constraint32(weights):
        return 1.11 - sum(weights)

    for errors in part2:
        initial_weights = np.full(len(errors), 1 / len(errors))
        bounds = [(0.02, 1.5) for _ in range(len(errors))]
        cons = [{'type': 'ineq', 'fun': constraint12, 'args': (errors,)},
                {'type': 'ineq', 'fun': constraint22, 'args': (errors,)},
                {'type': 'eq', 'fun': constraint32}]
        result = minimize(objective2, initial_weights, args=(errors,), method='SLSQP', bounds=bounds, constraints=cons)
        optimal_weights_list2.append(result.x)
    optimal_weights_array2 = np.array(optimal_weights_list2)

    ##########part3#########################
    optimal_weights_list3 = []

    def objective3(weights, errors):
        return np.abs(np.dot(weights, errors))

    def constraint13(weights, errors):
        return 25 - np.dot(weights, errors)

    def constraint23(weights, errors):
        return np.dot(weights, errors) + 25

    def constraint33(weights):
        return 1.22 - sum(weights)

    for errors in part3:
        initial_weights = np.full(len(errors), 1 / len(errors))
        bounds = [(0.07, 1.5) for _ in range(len(errors))]
        cons = [{'type': 'ineq', 'fun': constraint13, 'args': (errors,)},
                {'type': 'ineq', 'fun': constraint23, 'args': (errors,)},
                {'type': 'eq', 'fun': constraint33}]
        result = minimize(objective3, initial_weights, args=(errors,), method='SLSQP', bounds=bounds, constraints=cons)
        optimal_weights_list3.append(result.x)
    optimal_weights_array3 = np.array(optimal_weights_list3)

    reconstructed_data = []

    for i in range(10):  # Assuming 10 groups as in the original splitting
        start_idx = i * 10
        end_idx = start_idx + 10
        reconstructed_data.append(optimal_weights_array1[start_idx:end_idx, :])

        start_idx = i * 5
        end_idx = start_idx + 5
        reconstructed_data.append(optimal_weights_array2[start_idx:end_idx, :])

        start_idx = i * 9
        end_idx = start_idx + 9
        reconstructed_data.append(optimal_weights_array3[start_idx:end_idx, :])

    # Combining all the parts
    optimal_weights_array = np.vstack(reconstructed_data)
    optimal_weights_array = pd.DataFrame(optimal_weights_array)

    ###################구한 최적의 가중치랑 구하는 것##################
    a = pd.read_csv('evening_forecast.csv', parse_dates=True)
    a = a[a.columns[14:]]
    a.columns = optimal_weights_array.columns
    a.index = optimal_weights_array.index

    predamount = a * optimal_weights_array  ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    pred = predamount.values.tolist()
    pred = [item[0] for item in pred]

    return pred
def run(c,d):
    q = pd.read_csv('gens_by_sugi.csv', parse_dates=True)
    q = q[q.columns[1:]]
    q=q.values.tolist()
    q=[item[0] for item in q]
    pred1 = error_predictor10(c,d)
    pred2 = error_predictor17(c,d)
    pred1 = [round(num, 1) for num in pred1]
    pred2 = [round(num, 1) for num in pred2]
    adjusted_pred = []
    for x, y in zip(pred1, pred2):
        average = (x + y) / 2
        if 10 <= average <= 15.9:
            # Calculate the difference needed to add to y to make the average 15.9
            difference = (15.9 * 2) - x
            # Add this difference to the original y value
            adjusted_y = difference
        else:
            # If the average is not in the range, keep the original y value
            adjusted_y = y
        # Append the adjusted or original y value to the new list
        adjusted_pred.append(adjusted_y)
    pred2=adjusted_pred
    predamount = [(x + y) / 2 for x, y in zip(pred1, pred2)]
    predamount = [round(num, 1) for num in predamount]
    a = it.calc_profit(q, pred1)
    b = it.calc_profit(q, pred2)
    d = it.calc_profit(q, q)
    e = it.calc_profit(q, predamount)
    x = list(range(len(q)))
    plt.figure(figsize=(22, 14))
    for x_value in x:
        plt.axvline(x=x_value, color='lightgray', linestyle='--', linewidth=0.5)
    plt.plot(x, q,marker='o', color='black')
    #plt.plot(x, pred1, marker='o', color='green')
    plt.plot(x, pred2, marker='o', color='red')
    #plt.plot(x,predamount,marker='o',color='orange')

    for i, value in enumerate(pred2):
        plt.text(x[i], value, str(value), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, value in enumerate(q):
        plt.text(x[i], value, str(value), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    plt.title('Predictions Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # 그리드를 표시합니다.
    plt.grid(True)

            ##그래프를 화면에 보여줍니다.
    plt.show()
    g=it.calculate_mae(q,pred1)
    f=it.calculate_mae(q,pred2)
    h = it.calculate_mae(q, predamount)



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

    return g

# range_a = range(11)  # 0 ~ 10
# range_b = range(20)  # 0 ~ 15
range_c = range(6)  # 0 ~ 10
range_d = range(21)  # 0 ~ 15
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


