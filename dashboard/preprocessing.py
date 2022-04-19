from pandas import read_csv, to_datetime, concat
from numpy import where
from pickle import dump

# import data
holidays_data = read_csv('../data/holiday_17_18_19.csv - holiday_17_18_19.csv')
holidays = list(to_datetime(holidays_data["Date"], format='%d.%m.%Y'))
building_data_2017 = read_csv('../data/IST_South_Tower_2017_Ene_Cons.csv')
meteo_data = read_csv('../data/IST_meteo_data_2017_2018_2019.csv - IST_meteo_data_2017_2018_2019.csv')
building_data_2018 = read_csv('../data/IST_South_Tower_2018_Ene_Cons.csv')
building_data_2019 = read_csv('../data/IST_4buildings_2019.csv')

# process power data and concat
meteo_data['yyyy-mm-dd hh:mm:ss'] = to_datetime(meteo_data['yyyy-mm-dd hh:mm:ss'], format='%Y-%m-%d %H:%M:%S')
meteo_data = meteo_data.set_index('yyyy-mm-dd hh:mm:ss')
building_data_2017['Date_start'] = to_datetime(building_data_2017['Date_start'], format='%d-%m-%Y %H:%M')
building_data_2018['Date_start'] = to_datetime(building_data_2018['Date_start'], format='%d-%m-%Y %H:%M')
building_data_2019['Date_start'] = to_datetime(building_data_2019['Date'], format='%d-%m-%Y %H:%M')
building_data_2017 = building_data_2017.set_index('Date_start')
building_data_2018 = building_data_2018.set_index('Date_start')
building_data_2019 = building_data_2019.set_index('Date_start')
building_data_2019_south_tower = building_data_2019[['South Tower (kWh)']].copy()
building_data_2019_south_tower.columns = ['Power_kW']
building_data = concat([building_data_2017, building_data_2018, building_data_2019_south_tower])

# resample meteo data and join dataframes
meteo_data_resample = meteo_data.resample('H', closed='right', label='right')[
    'temp_C', 'HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar',
    'solarRad_W/m2', 'rain_mm/h'].mean()
meteo_data_resample["rain_day"] = meteo_data.resample('H', closed='right', label='right')['rain_day'].max()
df = building_data.join(meteo_data_resample)
df['holiday'] = where(df.index.to_period('D').astype('datetime64[ns]').isin(holidays), 1, 0)
df = df.fillna(0)

# add weekend information to dataframe
df['weekend'] = where(df.index.weekday > 4, 1, 0)

# extract date and time
df['hour'] = [date.hour for date in df.index]
df['day'] = [date.day for date in df.index]
df['month'] = [date.month for date in df.index]

# create feature of last day
df['Power_kW_-1_day'] = df['Power_kW'].shift(24)
df = df.fillna(0)

# Save result as pickle to reuse later in dashboard
with open('../data/dataframe_preprocessing.pickle', 'wb') as f:
    dump(df, f)

# Also save as csv for readability
df.to_csv('../data/dataframe_preprocessing.csv', index=True)


