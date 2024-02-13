import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model


def init_page():
    st.set_page_config(page_title='Weather Forecast',
                       page_icon="pictures/icon.png")

    st.markdown("""
            <style>
                .reportview-container { margin-top: -2em;}
                MainMenu {visibility: hidden;}
                .stDeployButton {display:none;}
                header {visibility: hidden;}
                footer {visibility: hidden;}
                stDecoration {display:none;}
            </style>
        """, unsafe_allow_html=True)
    st.header('Weather Forecast', divider='rainbow')


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


def app_start():
    init_page()
    if 'model_init' not in st.session_state:
        st.session_state['model_init'] = False

    clima_data = pd.read_csv('jena_climate_2009_2016.csv')
    df_clima = clima_data[['Date Time', 'T (degC)']].rename(columns={'Date Time': 'datetime', 'T (degC)': 'temperature'})
    df_clima['datetime'] = pd.to_datetime(df_clima['datetime'], format="%d.%m.%Y %H:%M:%S")
    df_clima_hour = df_clima[5::6]

    n_input = 20
    n_features = 1

    clima_model = load_model('LSTM_Models/lstm_univariate_btg.h5')

    with st.sidebar:
        st.header('Parameters', divider='rainbow')
        forecast_timeperiod = st.slider('Forecast days', 1, 48, 24)
        past_timeperiod = st.slider('Past days', 20, 8760, 240)
        period = st.slider('Period', 10000, 70000, 10000)

    # forecast_timeperiod = 24  # next 1 day
    # past_timeperiod = 240  # last 10 days
    predict_df = df_clima_hour[0:period]
    forecast_output = futureForecast(predict_df, 'temperature', n_input, n_features, forecast_timeperiod, clima_model)

    last_days = predict_df['temperature'][len(predict_df) - past_timeperiod:].tolist()

    next_days = pd.DataFrame(forecast_output, columns=['FutureForecast'])

    #if st.button("Show forecast"):
    st.line_chart(last_days, color="#FF0000")
    st.line_chart(next_days, color="#009900")


if __name__ == '__main__':
    app_start()
