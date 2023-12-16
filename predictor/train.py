import insentive as it
import pandas as pd
import numpy as np
import load as ld
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def post_10():
    test_x = it._get_weathers_forecasts10("2023-11-17")  ##api 로 내일 데이터 따오기
    test_x = test_x[test_x.columns[1:]]  ##시간 땐거

    error_data = ld.load_and_predict_model10(test_x)

    ##########나온 오차값들로 최적의 가중치를 조정하는 부분################
    # 결과를 저장할 리스트
    part1 = error_data[:10]
    part2 = error_data[10:15]
    part3 = error_data[15:]

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
        bounds = [(0.02, 1.5) for _ in range(len(errors))]
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
        bounds = [(0.03, 1.5) for _ in range(len(errors))]
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

    optimal_weights_array = np.concatenate([optimal_weights_array1, optimal_weights_array2, optimal_weights_array3],
                                           axis=0)
    optimal_weights_array = pd.DataFrame(optimal_weights_array)

    ###################구한 최적의 가중치랑 구하는 것##################
    a = it._get_gen_forecasts10('2023-11-17')  ##api 로 내일 모델 입찰량
    a = pd.DataFrame(a)
    print(a)
    a = a[a.columns[1:]]
    a.columns = optimal_weights_array.columns

    predamount = a * optimal_weights_array  ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    pred = predamount.values.tolist()
    pred = [item[0] for item in pred]

    x = list(range(24))
    plt.figure(figsize=(10, 6))

    plt.plot(x, pred, marker='o')

    for i, value in enumerate(pred):
        plt.text(x[i], value, str(value), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    plt.title('Predictions Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # 그리드를 표시합니다.
    plt.grid(True)

    # 그래프를 화면에 보여줍니다.
    plt.show()

    it._post_bids(pred)

    return pred
def post_17():
    test_x = it._get_weathers_forecasts17("2023-11-17")  ##api 로 내일 데이터 따오기
    test_x = test_x[test_x.columns[1:]]  ##시간 땐거

    error_data = ld.load_and_predict_model17(test_x)

    ##########나온 오차값들로 최적의 가중치를 조정하는 부분################
    # 결과를 저장할 리스트
    part1 = error_data[:10]
    part2 = error_data[10:15]
    part3 = error_data[15:]

    optimal_weights_list1 = []

    def objective1(weights, errors):
        return np.abs(np.dot(weights, errors))

    def constraint11(weights, errors):
        return 25 - np.dot(weights, errors)

    def constraint21(weights, errors):
        return np.dot(weights, errors) + 25

    def constraint31(weights):
        return 1.14 - sum(weights)

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
        return 1.1 - sum(weights)

    for errors in part2:
        initial_weights = np.full(len(errors), 1 / len(errors))
        bounds = [(0.03, 1.5) for _ in range(len(errors))]
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

    optimal_weights_array = np.concatenate([optimal_weights_array1, optimal_weights_array2, optimal_weights_array3],
                                           axis=0)
    optimal_weights_array = pd.DataFrame(optimal_weights_array)

    ###################구한 최적의 가중치랑 구하는 것##################
    a = it._get_gen_forecasts17('2023-11-17')  ##api 로 내일 모델 입찰량
    a = pd.DataFrame(a)
    print(a)
    a = a[a.columns[1:]]
    a.columns = optimal_weights_array.columns

    predamount = a * optimal_weights_array  ###가중치를 곱한 모델들을 섞은 예측발전량
    predamount = predamount.sum(axis=1).to_frame(name='predamounts')
    pred = predamount.values.tolist()
    pred = [item[0] for item in pred]
    # pred1 = post_10()
    # adjusted_pred = []
    # for x, y in zip(pred1, pred):
    #     average = (x + y) / 2
    #     if 5 <= average <= 15.9:
    #         difference = (15.9 * 2) - x
    #         adjusted_y = difference
    #     else:
    #         adjusted_y = y
    #     adjusted_pred.append(adjusted_y)
    # pred = adjusted_pred
    x = list(range(24))
    plt.figure(figsize=(10, 6))

    plt.plot(x, pred, marker='o')

    for i, value in enumerate(pred):
        plt.text(x[i], value, str(value), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    plt.title('Predictions Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # 그리드를 표시합니다.
    plt.grid(True)

    # 그래프를 화면에 보여줍니다.
    plt.show()

    pred=[0,0,0,0,0,0,0,3,
          16.3,30.1,26,32.7,20,29.8,23,17,
          15.9,0,0,0,0,0,0,0]
    it._post_bids(pred)


post_17()

