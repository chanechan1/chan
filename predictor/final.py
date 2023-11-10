import insentive as it
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib
def final_save_model10():
    a = pd.read_csv('pred.csv', parse_dates=True)  #
    a = a.iloc[:57840]
    a = a[a.columns[1:]]
    predictions1 = a.pivot( index='time' ,columns='model_id', values='amount')
    a = pd.read_csv('pred.csv', parse_dates=True)  #
    a = a.iloc[57840:]
    a = a[a.columns[1:]]
    predictions = a.pivot(index='time', columns='model_id', values='amount')

    predictions = pd.concat([predictions1, predictions], axis=0)
    actual_values = pd.read_csv('gens.csv', parse_dates=True)
    actual_values = actual_values[actual_values.columns[1:]]

    train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)
    #train_x = train_x.iloc[:11568]
    train_x = train_x[train_x.columns[2:]]  # 시간떼기

    predictions.index = train_x.index
    train_x = pd.concat([ train_x,predictions], axis=1)

    train_y = actual_values
    train_y = pd.concat([train_y, train_y], axis=0)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_x_values = train_x.values
    train_x_scaled = scaler_x.fit_transform(train_x_values)

    val_x_values = val_x.values
    val_x_scaled = scaler_x.transform(val_x_values)

    train_y_values = train_y.values
    train_y_scaled = scaler_y.fit_transform(train_y_values)

    train_x_reshaped = train_x_scaled.reshape((train_x_scaled.shape[0], 1, train_x_scaled.shape[1]))
    val_x_reshaped = val_x_scaled.reshape((val_x_scaled.shape[0], 1, val_x_scaled.shape[1]))

    input_shape = (1, train_x_scaled.shape[1])

    model = Sequential()
    # 입력 레이어와 LSTM 레이어 추가
    # LSTM 레이어의 첫 번째 인자는 뉴런의 수입니다.
    # input_shape는 (타임 스텝 수, 특징 수)로 설정해야 합니다.
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(
        train_x_reshaped,  # 학습 데이터
        train_y,  # 타겟 데이터
        epochs=350,  # 에포크 수: 전체 데이터셋에 대해 학습을 반복할 횟수
        batch_size=32,  # 배치 크기: 한 번에 네트워크에 전달되는 샘플의 수
        validation_data=(val_x_reshaped, val_y),  # 검증 데이터셋
        verbose=2 , # 학습 진행 상황의 표시 모드 (0: 출력 없음, 1: 진행 막대, 2: 에포크당 한 줄)
        callbacks=[early_stopping]
    )
    # 모델 저장
    model.save('ffinal_10')
    joblib.dump(scaler_x, 'scaler_x.joblib')

def final_model10(test_x):
    # 저장된 모델의 가중치를 로드
    model = load_model('ffinal_10')##저장됨
    # 새로운 데이터로 예측 수행
    scaler_x = joblib.load('scaler_x.joblib')
    test_x_scaled = scaler_x.transform(test_x.values)

    # 스케일링된 데이터를 모델이 요구하는 3차원 형태로 변환
    # 여기서 1은 각 샘플에 대한 타임 스텝 수를 가정한 값입니다.
    test_x_reshaped = test_x_scaled.reshape((test_x_scaled.shape[0], 1, test_x_scaled.shape[1]))

    predictions = model.predict(test_x_reshaped)

    return predictions


final_save_model10()

c=pd.read_csv('gens_by_sugi.csv',parse_dates=True)
c=c[c.columns[1:]]
c=c.values.tolist()
c = [item[0] for item in c]  #
test1=pd.read_csv('morning_forecast.csv',parse_dates=True)
test1=test1[test1.columns[1:]]
pred1=final_model10(test1)
pred1 = [item[0] for item in pred1]

a = it._get_gen_forecasts10('2023-11-09')  ##api 로 내일 모델 입찰량
a = pd.DataFrame(a)
a = a[a.columns[1:]]
pred=it._get_weathers_forecasts10('2023-11-09')
pred=pred[pred.columns[1:]]
pred.index = a.index
pred = pd.concat([pred,a], axis=1)
pred = final_model10(pred)
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
a=it.calc_profit(c,pred1)
a=sum(a)
print(a)

