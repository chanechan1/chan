import pandas as pd
import numpy as np



a = pd.read_csv('incentive.csv', parse_dates=True)  # 최적의 모델 찾기
arr=[]
########################가장 큰 인센티브를 나타내는
arr = []

arr = []

for hour in range(1, 25):
    # 합계를 저장할 딕셔너리 초기화
    h_sums = {}

    # 0번부터 4번 모델까지 반복
    for i in range(5):
        # 각 모델에 대한 데이터를 선택하고 합계를 구함
        # hour * 5는 각 시간대의 시작 행 인덱스를 계산
        h_sums[i] = a.iloc[(hour * 5) + i::120, 2].sum()

    # 딕셔너리의 항목을 값에 따라 오름차순으로 정렬
    sorted_h_sums = sorted(h_sums.items(), key=lambda x: x[1])

    # 최소값을 가진 두 개의 모델 인덱스를 찾음
    min_keys = [sorted_h_sums[0][0], sorted_h_sums[1][0]]

    # 결과 저장
    arr.append(min_keys)

    # 결과 출력
    print(f"{hour}시에 가장 작은 합계를 가진 모델 인덱스: {min_keys[0]}, {min_keys[1]}")

print(arr)
#각 시간마다 최대의 인센티브를 내는 모델을 사용 할 것
print('ㅁ')
