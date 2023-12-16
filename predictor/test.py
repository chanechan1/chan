import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import insentive as it
from sklearn.model_selection import train_test_split

train_x = pd.read_csv('weather_actual.csv', index_col=0, parse_dates=True)
train_y = pd.read_csv('gens.csv',index_col=0)
a  = it._get_weathers_forecasts10('2023-11-13')
a  = a[a.columns[1:]]

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

train_x_values = train_x.values
train_x_scaled = scaler_x.fit_transform(train_x_values)

train_y_values = train_y.values
train_y_scaled = scaler_y.fit_transform(train_y_values)

test_x_values = test_x.values
test_x_scaled = scaler_x.transform(test_x_values)

test_y_values = test_y.values
test_y_scaled = scaler_y.transform(test_y_values)


train_x_reshaped = train_x_scaled.reshape((train_x_scaled.shape[0], 1, train_x_scaled.shape[1]))
test_x_reshaped = test_x_scaled.reshape((test_x_scaled.shape[0], 1, test_x_scaled.shape[1]))


input_shape = (1, train_x_scaled.shape[1])

model = Sequential([
    Dense(256, activation='tanh', input_shape=(train_x.shape[1],)),
    Dense(320, activation='sigmoid'),
    Dense(1, activation='linear')
])
model.compile(loss='huber_loss', optimizer='adam')


history = model.fit(train_x, train_y, epochs=100, batch_size=32, validation_data=(test_x, test_y), verbose=2)

yhat = model.predict(a)


yhat = [item[0] for item in yhat]

plt.figure(figsize=(22, 14))
plt.plot(yhat, label='predict')
x = list(range(len(yhat)))

for i, value in enumerate(yhat):
    plt.text(x[i], value, str(value), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
plt.grid(True)

plt.show()
print
