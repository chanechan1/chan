import pandas as pd

# 예시 데이터프레임 생성
data1 = {'A': [1, 2, 3], }
data2 = {'A': [7, 8, 9], }

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# 데이터프레임 간의 요소별 곱셈
result = df1 * df2
print(result)
