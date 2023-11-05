
import insentive as it
import pandas as pd
import param as pa
import matplotlib.pyplot as plt
goodmodel=pa.goodmodel

b=it._get_gen_forecasts17() ##model 들의 예측량

selected_values = []

for i, model_index in enumerate(goodmodel):
    selected_value = b.iloc[i, model_index + 1]  # +1은 'time' 컬럼을 고려하여 인덱스를 조정
    selected_values.append(selected_value)

pred=selected_values

plt.figure(figsize=(15, 6))
plt.plot(pred, label='pred') #실제 발전량
plt.show()
print(b)

it._post_bids(pred)
