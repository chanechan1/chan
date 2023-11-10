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

# 학습률을 점진적으로 감소시키는 함수
def train_and_save_model10():

    a = pd.read_csv('pred.csv', parse_dates=True)  #
    a = a.iloc[:57840]
    a = a[a.columns[1:]]
    predictions1 = a.pivot(index='time', columns='model_id', values='amount')

    a = pd.read_csv('pred.csv', parse_dates=True)  #
    a = a.iloc[57840:]
    a = a[a.columns[1:]]
    predictions2 = a.pivot(index='time', columns='model_id', values='amount')

    predictions = pd.concat([predictions1, predictions2], axis=0)

    actual_values = pd.read_csv('gens.csv', parse_dates=True)
    actual_values = pd.concat([actual_values, actual_values], axis=0)

    actual_values = actual_values[actual_values.columns[1:]]
    errors = predictions.to_numpy() -actual_values.to_numpy().reshape(-1, 1)
    errors = pd.DataFrame(errors)

    train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)
    #train_x = train_x.iloc[:11568]
    train_x = train_x[train_x.columns[2:]]  # 시간떼기
    train_y = errors

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)
    model = Sequential([
        Dense(256, activation='tanh', input_shape=(train_x.shape[1],)),
        Dense(320, activation='sigmoid'),
        Dense(5, activation='linear')
    ])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
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
    model.save('fffinal_10')
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
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
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
    model.save('final_17')
def load_and_predict_model10(test_x):
    # 저장된 모델의 가중치를 로드
    model = load_model('final_10')##저장됨
    # 새로운 데이터로 예측 수행
    error_data = model.predict(test_x)
    return error_data
def load_and_predict_model17(test_x):
    # 저장된 모델의 가중치를 로드
    model = load_model('final_17')##저장됨
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
#     model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
#     early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
#     model.fit(
#         train_x, train_y,
#         epochs=300,
#         batch_size=32,
#         verbose=1,
#         validation_data=(val_x, val_y),
#         callbacks=[early_stopping]
#     )
#     # 모델의 가중치를 파일로 저장
#     model.save('final_10')
#train_and_save_model10()
