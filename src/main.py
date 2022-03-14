from pandas import read_csv, to_datetime, concat
from data_analysis import dimensionality
import matplotlib.pyplot as plt

# import data
holidays = list(read_csv('data/holiday_17_18_19.csv - holiday_17_18_19.csv')['Date'].values)

meteo_data = read_csv('data/IST_meteo_data_2017_2018_2019.csv - IST_meteo_data_2017_2018_2019.csv',
                      index_col='yyyy-mm-dd hh:mm:ss')
meteo_data.index = to_datetime(meteo_data.index, format='%Y-%m-%d %H:%M:%S')

power_north_tower_2017 = read_csv('data/IST_North_Tower_2017_Ene_Cons.csv - IST_North_Tower_2017_Ene_Cons.csv')
power_north_tower_2018 = read_csv('data/IST_North_Tower_2018_Ene_Cons.csv - IST_North_Tower_2018_Ene_Cons.csv')
power_north_tower_2017['Date_start'] = to_datetime(power_north_tower_2017['Date_start'], format='%d-%m-%Y %H:%M')
power_north_tower_2018['Date_start'] = to_datetime(power_north_tower_2018['Date_start'], format='%d-%m-%Y %H:%M')
power_north_tower_2017 = power_north_tower_2017.set_index('Date_start')
power_north_tower_2018 = power_north_tower_2018.set_index('Date_start')
power_north_tower = concat([power_north_tower_2017, power_north_tower_2018])

plt.figure()
plt.plot(power_north_tower.index, power_north_tower['Power_kW'])
plt.show()

dimensionality(power_north_tower)

