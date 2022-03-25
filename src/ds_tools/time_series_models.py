from statsmodels.tsa.arima.model import ARIMA
from matplotlib.pyplot import figure, savefig, show
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from pandas import DataFrame, to_datetime
import contextlib


def prophet_auto_ml(timestamp, prediction_variable, holidays=None, predict_period=0):
    holidays = DataFrame({
        'holiday': 'Portugal',
        'ds': holidays,
        'lower_window': 0,
        'upper_window': 1,
    })
    df = DataFrame()
    df['ds'] = timestamp
    df['y'] = prediction_variable
    model = Prophet(holidays=holidays, seasonality_mode='additive', yearly_seasonality=50)
    model.add_seasonality(name='yearly', period=10*24, fourier_order=5)
    model.fit(df)
    future = model.make_future_dataframe(periods=predict_period, freq='H')
    prediction = model.predict(future)
    model.plot(prediction, figsize=(24, 10))
    #savefig(f'{file_path}/prophet_auto_ml_{file_name}.png')
    model.plot_components(prediction)
    #savefig(f'{file_path}/prophet_auto_ml_components_{file_name}.png')
    prediction = prediction.set_index('ds')
    return prediction['yhat']


def arima_model(train, test):
    model = ARIMA(train, order=(1, 0, 1))
    model = model.fit()
    print(model.mae)
    prediction = model.predict(start=len(train) - 1, end=len(train) + len(test))
    figure(figsize=(24, 10))
    return prediction


def rendom_forrest_regressor(train, test):
    model = RandomForestRegressor().fit(train.array, test.array)

    return
