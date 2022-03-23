from pandas import read_csv, to_datetime, concat, date_range, DatetimeIndex
from ds_tools.profiling import show_dimensionality, show_outliers, show_distribution, show_sparsity, \
    show_histograms_numeric, plot_timeseries, plot_rolling_mean_dev, plot_seasonal_decompose, show_heatmap
from ds_tools.graphing.charts import plot_forecasting
from datetime import datetime
from numpy import where
from matplotlib.pyplot import figure, plot, show, legend, savefig
from ds_tools.time_series_models import arima_model, rendom_forrest_regressor, prophet_auto_ml
from ds_tools.utils import split_timeseries

PATH_GRAPHS = './images/profiling'

# DATA PREPARATION #

holidays_data = read_csv('./data/holiday_17_18_19.csv - holiday_17_18_19.csv')
holidays = list(to_datetime(holidays_data["Date"], format='%d.%m.%Y'))
meteo_data = read_csv('./data/IST_meteo_data_2017_2018_2019.csv - IST_meteo_data_2017_2018_2019.csv')
meteo_data['yyyy-mm-dd hh:mm:ss'] = to_datetime(meteo_data['yyyy-mm-dd hh:mm:ss'], format='%Y-%m-%d %H:%M:%S')
meteo_data = meteo_data.set_index('yyyy-mm-dd hh:mm:ss')

building_data_2017 = read_csv('./data/IST_South_Tower_2017_Ene_Cons.csv')
building_data_2018 = read_csv('./data/IST_South_Tower_2018_Ene_Cons.csv')
building_data_2017['Date_start'] = to_datetime(building_data_2017['Date_start'], format='%d-%m-%Y %H:%M')
building_data_2018['Date_start'] = to_datetime(building_data_2018['Date_start'], format='%d-%m-%Y %H:%M')
building_data_2017 = building_data_2017.set_index('Date_start')
building_data_2018 = building_data_2018.set_index('Date_start')

building_data = concat([building_data_2017, building_data_2018])

# resample data and combine into one dataframe
meteo_data_resample = meteo_data.resample('H', closed='left', label='right')[
    'temp_C', 'HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar',
    'solarRad_W/m2', 'rain_mm/h'].mean()
meteo_data_resample["rain_day"] = meteo_data.resample('H', closed='left', label='right')['rain_day'].max()
df = building_data.join(meteo_data_resample)
df['holiday'] = where(df.index.to_period('D').astype('datetime64[ns]').isin(holidays), True, False)

# data scaling

# DATA ANALYSIS #
start = datetime(year=2018, month=11, day=1)
end = datetime(year=2018, month=12, day=31)

# Print info
df.info()
show_dimensionality(df, file_path=PATH_GRAPHS, file_name='building')
# show_outliers(df, file_path=PATH_GRAPHS, file_name='building')
# show_distribution(df, file_path=PATH_GRAPHS, file_name='building')
# show_sparsity(df, file_path=PATH_GRAPHS, file_name='building')
# show_heatmap(df, file_path=PATH_GRAPHS, file_name='power')
# show_histograms_numeric(df, file_path=PATH_GRAPHS, file_name='building')
# plot_timeseries(df=df, columns=["temp_C", "Power_kW"], y_labels=['C', 'kW'], file_path=PATH_GRAPHS,
#                file_name='temp_power', start=start, end=end)
# plot_timeseries(df=df, columns=["Power_kW"], y_labels=['kW'], file_path=PATH_GRAPHS,
#                file_name='power', start=start, end=end)

# Rolling Means and Decomposition
# plot_rolling_mean_dev(df, column="Power_kW", file_path=PATH_GRAPHS, file_name='', window=24, start=start, end=end)
# plot_seasonal_decompose(df[start:end], column="Power_kW", file_path=PATH_GRAPHS, file_name='Power_kW')

# FEATURE ENGINEERING #
# add weekend information to dataframe
df['weekend'] = where(df.index.weekday > 4, True, False)
# select sliding window (drop data) or expanding window


# MODELING #
df_power = df['Power_kW']
train, test = split_timeseries(df_power, train_size=0.8)

# create auto-ml model for benchmarking - using prophet library from Meta in this case
prediction_prophet = prophet_auto_ml(timestamp=train.index, prediction_variable=train.values, holidays=holidays,
                                     file_path=PATH_GRAPHS, file_name='1', predict_period=len(test))
plot_forecasting(train=train, test=test, pred=prediction_prophet, x_label='time', y_label='kW', file_path=PATH_GRAPHS,
                 file_extension='1')

# create baseline regression model for benchmarking
prediction_arima = arima_model(train, test)
plot_forecasting(train=train, test=test, pred=prediction_arima, x_label='time', y_label='kW', file_path=PATH_GRAPHS,
                 file_extension='1')

# predict using regression models
prediction_forrest = rendom_forrest_regressor(train, test)

# predict using neural networks

# MODEL EVALUATION #
# split with forward chaining

# Alternative Strategy: Two different models: One for regular days and one for holidays
