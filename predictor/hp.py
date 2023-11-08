import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import tensorflow as tf
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=512, step=32),
                    activation=hp.Choice('activation1', values=['relu', 'tanh', 'sigmoid']),
                    input_shape=(train_x.shape[1],)))
    model.add(Dense(units=hp.Int('units2', min_value=32, max_value=512, step=32),
                    activation=hp.Choice('activation2', values=['relu', 'tanh', 'sigmoid'])))
    model.add(Dense(units=hp.Int('units3', min_value=32, max_value=512, step=32),
                    activation=hp.Choice('activation3', values=['relu', 'tanh', 'sigmoid'])))
    model.add(Dense(5, activation='linear'))  # 5개의 출력 가중치

    # 옵티마이저의 학습률을 튜닝
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='mean_squared_error',  # MSE는 회귀 문제에서 일반적으로 사용됩니다.
                  metrics=['mean_squared_error'])

    return model

# 하이퍼밴드 튜너를 인스턴스화
a = pd.read_csv('pred.csv', parse_dates=True)  #
a = a.iloc[:57840]
a = a[a.columns[1:]]
predictions = a.pivot(index='time', columns='model_id', values='amount')
actual_values = pd.read_csv('gens.csv', parse_dates=True)  ##
actual_values = actual_values[actual_values.columns[1:]]
errors = predictions.to_numpy() -actual_values.to_numpy().reshape(-1, 1)
errors = pd.DataFrame(errors)

train_x = pd.read_csv('weather_forecast.csv', parse_dates=True)
train_x = train_x.iloc[:11568]
train_x = train_x[train_x.columns[2:]]  # 시간떼기
train_y = errors

val_x=pd.read_csv('morning_forecast.csv')
val_x=val_x[val_x.columns[1:14]]
b=pd.read_csv('gens_by_sugi.csv')
b=b[b.columns[1:]]
a=pd.read_csv('morning_forecast.csv')
a=a[a.columns[14:]]
val_y=a.to_numpy()-b.to_numpy().reshape(-1,1)

tuner = kt.Hyperband(build_model,
                     objective='val_mean_squared_error',
                     max_epochs=10,
                     factor=3,  # 감소율(decay rate) 설정
                     directory='my_dir',
                     project_name='intro_to_kt')

# 조기 종료 콜백 정의
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# 튜너를 사용한 하이퍼파라미터 탐색
tuner.search(train_x, train_y,
             epochs=50,
             validation_data=(val_x, val_y),
             callbacks=[stop_early])

# 최적의 하이퍼파라미터를 가져옴
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# 최적의 하이퍼파라미터로 모델을 다시 구축
model = tuner.hypermodel.build(best_hps)

# 최적화된 하이퍼파라미터로 모델을 훈련
history = model.fit(train_x, train_y,
                    epochs=50,
                    validation_data=(val_x, val_y))

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units1')}, the optimal activation is {best_hps.get('activation1')},
the optimal number of units in the second densely-connected layer is {best_hps.get('units2')},
the optimal activation is {best_hps.get('activation2')}, and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
