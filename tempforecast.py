import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import date

from keras.models import Sequential, save_model
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam


def input_LSTM(df, input_sequence):

    df_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_np) - input_sequence):
        row = [a for a in df_np[i:i + input_sequence]]
        X.append(row)
        label = df_np[i + input_sequence]
        y.append(label)

    return np.array(X), np.array(y)


clim_data = pd.read_csv('jena_climate_2009_2016.csv')
print(clim_data.head())

df_clim = clim_data[['Date Time', 'T (degC)']].rename(columns={'Date Time': 'datetime', 'T (degC)': 'temperature'})
df_clim['datetime'] = pd.to_datetime(df_clim['datetime'], format="%d.%m.%Y %H:%M:%S")

df_clim_hour = df_clim[5::6]

# Create the Train-Test Split

n_input = 20

df_min_model_data = df_clim_hour['temperature']

X, y = input_LSTM(df_min_model_data, n_input)

# Training data
X_train, y_train = X[:60000], y[:60000]

# Validation data
X_val, y_val = X[60000:65000], y[60000:65000]

# Test data
X_test, y_test = X[65000:], y[65000:]

# Create the LSTM Model

n_features = 1

model1 = Sequential()
model1.add(InputLayer((n_input, n_features)))
model1.add(LSTM(100, return_sequences=True))
model1.add(LSTM(100, return_sequences=True))
#model1.add(LSTM(50))
model1.add(LSTM(64))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(1, activation='linear'))

# model1.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=2)

model1.compile(loss=MeanSquaredError(),
               optimizer=Adam(learning_rate=0.0001),
               metrics=RootMeanSquaredError())

model1.fit(X_train, y_train,
           validation_data=(X_val, y_val),
           epochs=50,
           callbacks=[early_stop])

# Check the model performance

losses_df1 = pd.DataFrame(model1.history.history)
losses_df1.plot(figsize=(10, 6))
plt.show()

# Safe model

# save the model
save_model(model1, "LSTM_Models/lstm_forecast_model.h5")


