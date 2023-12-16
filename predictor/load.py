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
from keras.models import load_model

#학습률을 점진적으로 감소시키는 함수
def train_and_save_model10():

    a = pd.read_csv('pred.csv',index_col=0, parse_dates=True)  #
    a = a.iloc[:57840]
    predictions1 = a.pivot(index='time', columns='model_id', values='amount')

    a = pd.read_csv('pred.csv',index_col=0 , parse_dates=True)  #
    a = a.iloc[57840:]
    predictions2 = a.pivot(index='time', columns='model_id', values='amount')

    predictions = pd.concat([predictions1, predictions2], axis=0)

    # indices = []
    # for i in range(0, len(predictions), 24):
    #     start = i + 7
    #     end = i + 18
    #     indices.extend(range(start, end + 1))
    # predictions = predictions.iloc[indices]


    actual_values = pd.read_csv('gens.csv', parse_dates=True)
    actual_values = pd.concat([actual_values, actual_values], axis=0)
    # indices = []
    # for i in range(0, len(actual_values), 24):
    #     start = i + 7
    #     end = i + 18
    #     indices.extend(range(start, end + 1))
    # actual_values = actual_values.iloc[indices]
    actual_values = actual_values[actual_values.columns[1:]]
    errors = predictions.to_numpy() -actual_values.to_numpy().reshape(-1, 1)
    errors = pd.DataFrame(errors)

    train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)
    # indices = []
    # for i in range(0, len(train_x), 24):
    #     start = i + 7
    #     end = i + 18
    #     indices.extend(range(start, end + 1))
    # train_x = train_x.iloc[indices]
    train_x = train_x[train_x.columns[2:]]  # 시간떼기
    train_y = errors

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)
    model = Sequential([
        Dense(256, activation='tanh', input_shape=(train_x.shape[1],)),
        Dense(320, activation='sigmoid'),
        Dense(5, activation='linear')
    ])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='huber_loss', optimizer=optimizer, metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(
        train_x, train_y,
        epochs=600,
        batch_size=32,
        verbose=1,
        validation_data=(val_x, val_y),
        callbacks=[early_stopping]
    )
    # 모델의 가중치를 파일로 저장
    model.save('huberlastfinal')
def train_and_save_model17():

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

    model = Sequential([
        Dense(256, activation='tanh', input_shape=(train_x.shape[1],)),
        Dense(320, activation='sigmoid'),
        Dense(5, activation='linear')
    ])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(
        train_x, train_y,
        epochs=600,
        batch_size=32,
        verbose=1,
        validation_data=(val_x, val_y),
        callbacks=[early_stopping]
    )
    # 모델의 가중치를 파일로 저장
    model.save('11568mae_17')
def load_and_predict_model10(test_x):
    # 저장된 모델의 가중치를 로드
    model = load_model('huberfinal')##저장됨
    # 새로운 데이터로 예측 수행
    error_data = model.predict(test_x)
    return error_data
def load_and_predict_model17(test_x):
    # 저장된 모델의 가중치를 로드
    model = load_model('huberfinal')##저장됨
    # 새로운 데이터로 예측 수행
    error_data = model.predict(test_x)
    return error_data

# def train_and_save_model10():
#
#     a = pd.read_csv('pred.csv', parse_dates=True)  #
#     a = a.iloc[:57840]
#     a = a[a.columns[1:]]
#     predictions = a.pivot(index='time', columns='model_id', values='amount')
#     actual_values = pd.read_csv('gens.csv', parse_dates=True)
#     actual_values = actual_values[actual_values.columns[1:]]
#     errors = predictions.to_numpy() -actual_values.to_numpy().reshape(-1, 1)
#     errors = pd.DataFrame(errors)
#
#     train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)
#     train_x = train_x.iloc[:11568]
#     train_x = train_x[train_x.columns[2:]]  # 시간떼기
#     train_y = errors
#
#     train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)
#     model = Sequential([
#         Dense(256, activation='tanh', input_shape=(train_x.shape[1],)),
#         Dense(320, activation='sigmoid'),
#         Dense(5, activation='linear')
#     ])
#     optimizer = Adam(learning_rate=0.0001)
#     model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
#     early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
#     model.fit(
#         train_x, train_y,
#         epochs=600,
#         batch_size=32,
#         verbose=1,
#         validation_data=(val_x, val_y),
#         callbacks=[early_stopping]
#     )
#     # 모델의 가중치를 파일로 저장
#     model.save('11568mae_10')
#train_and_save_model10()
