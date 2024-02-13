import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, save_model, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam


def futureForecast(df, col, n_input, n_features, forecast_timeperiod, model):
    x_input = np.array(df[len(df) - n_input:][col])
    temp_input = list(x_input)
    lst_output = []
    i = 0

    while i < forecast_timeperiod:
        if len(temp_input) > n_input:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape((n_input, n_features))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input = temp_input[1:]
            lst_output.append(yhat[0][0])

            i = i + 1

        else:
            x_input = x_input.reshape((n_input, n_features))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])

            i = i + 1

    return lst_output


def sequential_input_LSTM(df, input_sequence):
    df_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_np) - input_sequence):
        row = [a for a in df_np[i:i + input_sequence]]
        X.append(row)
        label = df_np[i + input_sequence]
        y.append(label)

    return np.array(X), np.array(y)


def app_start():
    clim_data = pd.read_csv('jena_climate_2009_2016.csv')

    df_clim = clim_data[['Date Time', 'T (degC)']].rename(columns={'Date Time': 'datetime', 'T (degC)': 'temperature'})
    df_clim['datetime'] = pd.to_datetime(df_clim['datetime'], format="%d.%m.%Y %H:%M:%S")

    df_clim_hour = df_clim[5::6]
    print(len(df_clim_hour))

    # Create the Train-Test Split

    n_input = 20

    df_min_model_data = df_clim_hour['temperature']

    X, y = sequential_input_LSTM(df_min_model_data, n_input)

    # Training data
    X_train, y_train = X[:60000], y[:60000]

    # Validation data
    X_val, y_val = X[60000:65000], y[60000:65000]

    # Test data
    X_test, y_test = X[65000:], y[65000:]

    # load the model
    model1 = load_model('LSTM_Models/lstm_forecast_model.h5')

    # Predict the temperature against the test data
    test_predictions1 = model1.predict(X_test).flatten()

    X_test_list = []
    for i in range(len(X_test)):
        X_test_list.append(X_test[i][0])

    test_predictions_df1 = pd.DataFrame({'X_test': list(X_test_list), 'LSTM Prediction': list(test_predictions1)})

    # LSTM temperature forecast on complete Test Data

    # test_predictions_df1.plot(figsize=(15, 6))

    # LSTM temperature forecast on last 1 Month in the Test Data (720 hours)

    #test_predictions_df1[(len(X_test) - 720):].plot(figsize=(15, 5))

    #train_results = pd.DataFrame(data={'Train Predictions': list(test_predictions1), 'Actuals': list(y_test)})
    #print(train_results[50:100])
    #plt.plot(train_results['Train Predictions'][len(train_results)-240:])
    #plt.plot(train_results['Actuals'][len(train_results)-240:])


    n_input = 20
    n_features = 1
    forecast_timeperiod = 24        # next 1 days
    past_timeperiod = 240  # last 10 days
    model = model1

    predict_df = df_clim_hour[5000:10000]

    forecast_output = futureForecast(predict_df, 'temperature', n_input, n_features, forecast_timeperiod, model)

    last_10_days = predict_df['temperature'][len(predict_df) - past_timeperiod:].tolist()


    next_10_days = pd.DataFrame(forecast_output, columns=['FutureForecast'])

    plt.figure(figsize=(15, 5))

    hist_axis = len(last_10_days)
    forecast_axis = hist_axis + len(next_10_days)

    plt.plot(np.arange(0, hist_axis), last_10_days, color='blue')
    plt.plot(np.arange(hist_axis, forecast_axis), next_10_days['FutureForecast'].tolist(), color='orange')

    plt.title('LSTM Forecast for Next 10 Days')
    plt.xlabel('Hours')
    plt.ylabel('Temperature')
    plt.show()
    # save the figure
    # plt.savefig('Pics_Models/lstm_univariate_forecast_pic.png')
    # plt.savefig('Pics_Models/lstm_univariate_forecast_pdf.pdf')


if __name__ == '__main__':
    app_start()
