import insentive as it
import pandas as pd
import numpy as np
import load as ld
from scipy.optimize import minimize
import matplotlib.pyplot as plt

test_x = it._get_weathers_forecasts10("2023-11-11")  ##api 로 내일 데이터 따오기
test_x = test_x[test_x.columns[1:]]  ##시간 땐거

error_data = ld.load_and_predict_model10(test_x)

##########나온 오차값들로 최적의 가중치를 조정하는 부분################
# 결과를 저장할 리스트

optimal_weights_list = []


# 목적 함수
def objective(weights, errors):
    return np.abs(np.dot(weights, errors))


# 제약 조건 함수들
def constraint1(weights, errors):
    return 6 - np.dot(weights, errors)


def constraint2(weights, errors):
    return np.dot(weights, errors) + 6


def constraint3(weights):
    return 1.18 - sum(weights)


# 각 행에 대해 최적화를 수행하는 루프
for errors in error_data:
    # 초기 가중치
    initial_weights = np.full(len(errors), 1 / len(errors))

    # bounds를 각 행의 오차 수에 맞게 설정
    bounds = [(0.1, 1) for _ in range(len(errors))]

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
a = it._get_gen_forecasts10('2023-11-11')  ##api 로 내일 모델 입찰량
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

plt.plot(x,pred,marker='o')

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
