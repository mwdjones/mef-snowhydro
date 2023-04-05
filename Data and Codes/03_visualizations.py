# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:09:00 2023


@author: M.W.Jones
"""
#%%

'''PACKAGE IMPORTS'''
#For data analysis
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
import scipy
import xarray as xr

#%%
'''Import Data'''
import_path = "D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Cleaned Data/"
import_path_frost = "D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Raw Data/"


### Import Snow Data
s2data = xr.open_dataset(import_path + '01_cleanedsnowdataS2.nc')
s2data_df = s2data.to_dataframe().reset_index(drop = False)
s2data_df = s2data_df.replace('NaN', np.nan)
s2data_df = s2data_df.replace('nan', np.nan)
s2data_df.time = pd.to_datetime(s2data_df.time)
s2data_df['watershed'] = 'S2'

s6data = xr.open_dataset(import_path + '01_cleanedsnowdataS6.nc')
s6data_df = s6data.to_dataframe().reset_index(drop = False)
s6data_df = s6data_df.replace('NaN', np.nan)
s6data_df = s6data_df.replace('nan', np.nan)
s6data_df.time = pd.to_datetime(s6data_df.time)
s6data_df['watershed'] = 'S6'


### Import Frost Data
all_frost = pd.read_csv(import_path_frost + 'Snow/mef_snowfrost_data.csv',
    parse_dates = ['DATE'], 
    na_values = ['NaN'])

#Format Data
all_frost = all_frost[1:]
frost = all_frost.dropna(subset = ['FROST'])
frost = frost.replace('2023-02-01', '2023-01-30')
frost = frost[['STAKE NO', 'DATE', 'FROST.1']]
frost['STAKE NO'] =[col.strip() for col in frost['STAKE NO']]
frost['FROST.1'] =[np.float64(c) for c in frost['FROST.1']]
frost['Zones'] = np.arange(0, len(frost.DATE))
frost['Watershed'] = np.arange(0, len(frost.DATE))

#Sort Stake Names
for idx,row in frost.iterrows():
    stake = row[0]

    if stake.startswith('S6'):
        frost.Watershed[idx] = 'S6'
    else:
        frost.Watershed[idx] = 'S2'
        
    if stake in ['S603', 'S632', 'S04', 'S05', 'S15', 'S44', 'S54']:
        frost.Zones[idx] = 'Upland'
    elif stake in ['S613', 'S622', 'S25']:
        frost.Zones[idx] = 'Lagg'
    else:
        frost.Zones[idx] = 'Bog'

### Import Soil Moisture and Temperature Data
soilData = pd.read_csv(import_path + '01_cleanedsensordata.csv', 
                       parse_dates= ['DateTime'], 
                       na_values = ['NaN'])

#Precipitation data - update from MN DNR site occasionally, eventually replace with MEF data
precip_directory = 'D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Raw Data/'
precip = pd.read_csv(precip_directory + 'GrandRapids_Precip_MNDNR.csv', 
                     na_values = ['T', 'M'], 
                     parse_dates = ['Date'], 
                     names = ['Date', 'Tmax_F', 'Tmin_F', 'P_in', 'Snow_in', 'SnowDepth_in'], 
                     header = 0, 
                     dtype = {'P_in':float, 'Snow_in':float, 'SnowDepth_in':float})


#%%
'''FUNCTIONS'''

def fillFrostSeries(s, beginDate, endDate):
    range = pd.date_range(start = beginDate.date(), end = endDate.date())
    stakes = list(set(s['STAKE NO']))

    #Upsample frost series - prevents random zeroes
    #s = s.set_index(['DATE', 'STAKE NO']).resample('D').ffill().reset_index()

    #Index for resampling
    replaceInd = pd.MultiIndex.from_arrays([np.repeat(range, len(stakes)), np.tile(stakes, len(range))])
    
    s_mod = s.set_index(['DATE', 'STAKE NO']).reindex(replaceInd).rename_axis(['DATE', 'STAKE NO'])
    
    return s_mod.unstack().fillna(method = 'pad').stack().reset_index()

def snipPrecip(P, fir, la):
    #function trims the precip log to the specified dates, filling in NaNs where there is no available data
    #P must be a timeseries containing at least two columns, one with dates and one with precip values
    range = pd.date_range(start = fir.date(), end = la.date())
    return P.set_index('Date').reindex(range).rename_axis('Date').reset_index()
   

#%%
'''SNOW AND FROST TIME SERIES'''

ylimit = 60
ylimit_frost = 25

def plotSnowSeries(gsx, gsy, data, frostData, color, xticks = False):
    ax = plt.subplot(gs[gsx, gsy])

    #All snow series
    sns.lineplot(x = data.time,
                y = data.depths, 
                ax = ax, 
                estimator = None, 
                units = data.stakes, 
                color = 'silver', 
                lw = 1
                )

    #Mean snow series
    sns.lineplot(x = data.time,
                y = data.depths, 
                ax = ax,  
                ci = None, 
                color = color)
    
    #Secondary Axis - Frost
    ax2 = ax.twinx()

    sns.lineplot(x = frostData.DATE, 
                 y = frostData['FROST.1'], 
                 color = color, 
                 ci = None,
                 ax = ax2)
    line = plt.gca().lines
    plt.fill_between(line[0].get_xdata(), 0, line[0].get_ydata(),
                     color = color, 
                     alpha = 0.4)
    
    
    #hline
    ax.axhline(y = 0, xmin = 0, xmax = 1, color = 'silver')

    #Plot settings
    ax.set_ylabel(' ')
    if (not xticks):
       ax.tick_params(
           axis = 'x', 
           which = 'both', 
           bottom = False, 
           top = False, 
           labelbottom = False
       )
       ax.set_xlabel(' ')
    ax.set_xlim(min(data.time), max(data.time))
    ax.set_ylim(-ylimit, ylimit)

    ax2.set_ylabel(' ')
    ax2.set_ylim(-ylimit_frost, ylimit_frost)
    ax2.invert_yaxis()

### S2
gs = GridSpec(3, 1, wspace = 0.40, hspace = 0.20)
#Axis 1 - Upland
snowTemp_Upland = s2data_df[s2data_df.zones == 'Upland']
frost_Upland = frost[(frost.Watershed == 'S2') & (frost.Zones == 'Upland')]
frost_UplandFilled = fillFrostSeries(frost_Upland, min(snowTemp_Upland.time), max(snowTemp_Upland.time))
#frost_UplandFilled['FROST.1'] = frost_UplandFilled['FROST.1'].fillna(0)
plotSnowSeries(0, 0, snowTemp_Upland, frost_UplandFilled, '#1b9e77')
#Axis 2 - Lagg
snowTemp_Lagg = s2data_df[s2data_df.zones == 'Lagg']
frost_Lagg = frost[(frost.Watershed == 'S2') & (frost.Zones == 'Lagg')]
frost_LaggFilled = fillFrostSeries(frost_Lagg, min(snowTemp_Lagg.time), max(snowTemp_Lagg.time))
plotSnowSeries(1, 0, snowTemp_Lagg, frost_LaggFilled, '#7570b3')
#Axis 3 - Bog
snowTemp_Bog = s2data_df[s2data_df.zones == 'Bog']
frost_Bog = frost[(frost.Watershed == 'S2') & (frost.Zones == 'Bog')]
frost_BogFilled = fillFrostSeries(frost_Bog, min(snowTemp_Bog.time), max(snowTemp_Bog.time))
plotSnowSeries(2, 0, snowTemp_Bog, frost_BogFilled, '#d95f02', xticks = True)

plt.suptitle('S2')
plt.xticks(rotation=30)
plt.show()


### S6
gs = GridSpec(3, 1, wspace = 0.40, hspace = 0.20)
#Axis 1 - Upland
snowTemp_Upland = s6data_df[s6data_df.zones == 'Upland']
frost_Upland = frost[(frost.Watershed == 'S6') & (frost.Zones == 'Upland')]
frost_UplandFilled = fillFrostSeries(frost_Upland, min(snowTemp_Upland.time), max(snowTemp_Upland.time))
#frost_UplandFilled['FROST.1'] = frost_UplandFilled['FROST.1'].fillna(0)
plotSnowSeries(0, 0, snowTemp_Upland, frost_UplandFilled, '#1b9e77')
#Axis 2 - Lagg
snowTemp_Lagg = s6data_df[s6data_df.zones == 'Lagg']
frost_Lagg = frost[(frost.Watershed == 'S6') & (frost.Zones == 'Lagg')]
frost_LaggFilled = fillFrostSeries(frost_Lagg, min(snowTemp_Lagg.time), max(snowTemp_Lagg.time))
plotSnowSeries(1, 0, snowTemp_Lagg, frost_LaggFilled, '#7570b3')
#Axis 3 - Bog
snowTemp_Bog = s6data_df[s6data_df.zones == 'Bog']
frost_Bog = frost[(frost.Watershed == 'S6') & (frost.Zones == 'Bog')]
frost_BogFilled = fillFrostSeries(frost_Bog, min(snowTemp_Bog.time), max(snowTemp_Bog.time))
plotSnowSeries(2, 0, snowTemp_Bog, frost_BogFilled, '#d95f02', xticks = True)

plt.suptitle('S6')
plt.xticks(rotation=30)
plt.show()


# %%

'''SOIL MOISTURE and LOCATION'''
soilData['Watershed'] = np.arange(0, len(soilData.DateTime))
soilData['Orientation'] = np.arange(0, len(soilData.DateTime))

#Sort Stakes
for idx,row in soilData.iterrows():
    stake = row['SensorName']

    if stake.startswith('S6'):
        soilData.Watershed[idx] = 'S6'
    else:
        soilData.Watershed[idx] = 'S2'
        
    if stake in ['S603', 'S613', 'S244', 'S254']:
        soilData.Orientation[idx] = 'South'
    else:
        soilData.Orientation[idx] = 'North'

### Plot
fig, [ax1, ax2] = plt.subplots(2, 1, constrained_layout = True, 
    figsize = (8,6))

#Soil Moisture
sns.lineplot(data = soilData,
             x = 'DateTime', y = 'SoilMoist_15cm', 
             #units = 'SensorName', 
             #estimator = None,
             hue = 'Watershed', 
             style = 'Orientation', 
             dashes = True, markers = False, 
             ax = ax1)
Psnip = snipPrecip(precip, min(soilData.DateTime), max(soilData.DateTime))
ax1.bar(Psnip.Date, Psnip.P_in, color = 'lightgrey', zorder = -2)

ax1.set_xlim(min(soilData.DateTime), max(soilData.DateTime))
ax1.set_ylim(0, 50)
ax1.set_ylabel('Soil Moisture at 15cm depth')
ax1.set_xlabel('Date')

#Soil Temperature
sns.lineplot(data = soilData,
             x = 'DateTime', y = 'SoilTemp1', 
             #units = 'SensorName', 
             #estimator = None,
             hue = 'Watershed', 
             style = 'Orientation', 
             dashes = True, markers = False, 
             ax = ax2)

ax2.bar(Psnip.Date, Psnip.P_in, color = 'lightgrey', zorder = -2)

ax2.set_xlim(min(soilData.DateTime), max(soilData.DateTime))
ax2.set_ylim(0, 14)
ax2.set_ylabel('Soil Temperature at 5cm depth')
ax2.set_xlabel('Date')

# %%
