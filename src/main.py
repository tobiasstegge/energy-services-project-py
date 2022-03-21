from pandas import read_csv, to_datetime, concat, DataFrame
from ds_tools.profiling import show_dimensionality, show_outliers, show_distribution, show_sparsity, \
    show_histograms_numeric, plot_timeseries, plot_rolling_mean_dev, plot_seasonal_decompose, show_heatmap
import datetime

PATH_GRAPHS = './images/profiling'
PATH_METEO_DATA = './data/IST_meteo_data_2017_2018_2019.csv - IST_meteo_data_2017_2018_2019.csv'
PATH_HOLIDAY_DATA = './data/holiday_17_18_19.csv - holiday_17_18_19.csv'
PATH_BUILDING_DATA_2017 = './data/IST_South_Tower_2017_Ene_Cons.csv'
PATH_BUILDING_DATA_2018 = './data/IST_South_Tower_2018_Ene_Cons.csv'

TARGET = 'Power_kW'

# DATA PREPARATION #

holiday_data = read_csv(PATH_HOLIDAY_DATA)
holiday_data["Date"] = to_datetime(holiday_data["Date"], format='%d.%m.%Y')
holiday_data = holiday_data.set_index('Date')

meteo_data = read_csv(PATH_METEO_DATA, index_col='yyyy-mm-dd hh:mm:ss')
meteo_data.index = to_datetime(meteo_data.index, format='%Y-%m-%d %H:%M:%S')

building_data_2017 = read_csv(PATH_BUILDING_DATA_2017)
building_data_2018 = read_csv(PATH_BUILDING_DATA_2018)
building_data_2017['Date_start'] = to_datetime(building_data_2017['Date_start'], format='%d-%m-%Y %H:%M')
building_data_2018['Date_start'] = to_datetime(building_data_2018['Date_start'], format='%d-%m-%Y %H:%M')
building_data_2017 = building_data_2017.set_index('Date_start')
building_data_2018 = building_data_2018.set_index('Date_start')

power_data = concat([building_data_2017, building_data_2018])

# resample meteo data
meteo_data_resample = meteo_data.resample('H', closed='left', label='right')[
    'temp_C', 'HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar',
    'solarRad_W/m2', 'rain_mm/h'].mean()
meteo_data_resample["rain_day"] = meteo_data.resample('H', closed='left', label='right')['rain_day'].max()

# combine data into one dataframe
df = power_data.join(meteo_data_resample)

### DATA ANALYSIS ###
start = datetime.datetime(year=2018, month=12, day=1)
end = datetime.datetime(year=2018, month=12, day=31)

# Print info
df.info()
#show_dimensionality(df, file_path=PATH_GRAPHS, file_name='building')
#show_outliers(df, file_path=PATH_GRAPHS, file_name='building')
#show_distribution(df, file_path=PATH_GRAPHS, file_name='building')
#show_sparsity(df, file_path=PATH_GRAPHS, file_name='building')
show_heatmap(df, file_path=PATH_GRAPHS, file_name='power')
show_histograms_numeric(df, file_path=PATH_GRAPHS, file_name='building')
plot_timeseries(df=df, columns=["temp_C", "Power_kW"], y_labels=['C', 'kW'], file_path=PATH_GRAPHS,
                file_name='temp_power', start=start, end=end)
plot_timeseries(df=df, columns=["Power_kW"], y_labels=['kW'], file_path=PATH_GRAPHS,
                file_name='power', start=start, end=end)

# Rolling Means and Decomposition
plot_rolling_mean_dev(df, column="Power_kW", file_path=PATH_GRAPHS, file_name='', window=24, start=start, end=end)
plot_seasonal_decompose(df[start:end], column="Power_kW", file_path=PATH_GRAPHS, file_name='Power_kW')

# MODELING #


# MODEL EVALUATION #

# Alternative Strategy: Two different models: One for regular days and one for holidays
