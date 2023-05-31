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
import statsmodels.api as sm

#%%
'''IMPORT DATA'''
import_path = "D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Cleaned Data/"
import_path_raw = "D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Raw Data/"

### Import Snow Data
s2data = xr.open_dataset(import_path + '01_cleanedsnowdataS2.nc')
s2data_df = s2data.to_dataframe().reset_index(drop = False)
s2data_df = s2data_df.replace('NaN', np.nan)
s2data_df = s2data_df.replace('nan', np.nan)
s2data_df.time = pd.to_datetime(s2data_df.time)
s2data_df['watershed'] = 'S2'
#Save separately
s2data_df.to_csv(import_path + '01_cleanedsnowdataS2.csv')

s6data = xr.open_dataset(import_path + '01_cleanedsnowdataS6.nc')
s6data_df = s6data.to_dataframe().reset_index(drop = False)
s6data_df = s6data_df.replace('NaN', np.nan)
s6data_df = s6data_df.replace('nan', np.nan)
s6data_df.time = pd.to_datetime(s6data_df.time)
s6data_df['watershed'] = 'S6'
s6data_df.to_csv(import_path + '01_cleanedsnowdataS6.csv')


### Import Frost Data
all_frost = pd.read_csv(import_path_raw + 'Snow/mef_snowfrost_data.csv',
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

### Import LAI Data
s2LAI_import = pd.read_csv(import_path_raw + "S2_winterLAI.txt", sep = ';')
s6LAI_import = pd.read_csv(import_path_raw + "S6_summerLAI.txt", sep = ';')

#Trim colnames
s2LAI_import.columns =[col.strip() for col in s2LAI_import.columns]
s6LAI_import.columns =[col.strip() for col in s6LAI_import.columns]

#Subset columns of interest
nameDict = {"User Field 1":"Stake", "User Field 2":"Orientation", "User Field 3":"Zone"}
s2LAI = s2LAI_import[["User Field 1", "User Field 2", "User Field 3", "% Sky Area", "% Mask Area", "% Cnpy Open",
    "% Site Open", "LAI 4Ring", "LAI 5Ring", "Date", "Time"]]
s2LAI = s2LAI.rename(columns = nameDict)
s6LAI = s6LAI_import[["User Field 1", "User Field 2", "User Field 3", "% Sky Area", "% Mask Area", "% Cnpy Open", 
    "% Site Open", "LAI 4Ring", "LAI 5Ring", "Date", "Time"]]
s6LAI = s6LAI.rename(columns = nameDict)

#Separate stake values into new column
s2LAI['Stake_ID'] = [name[0:4] for name in s2LAI.Stake]
s6LAI['Stake_ID'] = [name[0:4] for name in s6LAI.Stake]

###Groupby
#Group numerical values
s2LAI_groupednums = pd.DataFrame(s2LAI.groupby(["Stake_ID"], as_index = False).mean()) #Takes the mean of each variable, drops Date, Time, Orientation, and Zone because they are not ints
s6LAI_groupednums = pd.DataFrame(s6LAI.groupby(["Stake_ID"], as_index = False).mean()) #Takes the mean of each variable, drops Date, Time, Orientation, and Zone because they are not ints
#Group string values
s2LAI_groupednames = pd.DataFrame(s2LAI.groupby(["Stake_ID"], as_index = False)['Zone'].max())
s6LAI_groupednames = pd.DataFrame(s6LAI.groupby(["Stake_ID"], as_index = False)['Zone'].max())
#Merge
s2LAI_grouped = s2LAI_groupednames.merge(s2LAI_groupednums, how = 'outer')
s6LAI_grouped = s6LAI_groupednames.merge(s6LAI_groupednums, how = 'outer')

#Trim zone names
s2LAI_grouped.Zone =[col.strip() for col in s2LAI_grouped.Zone]
s6LAI_grouped.Zone =[col.strip() for col in s6LAI_grouped.Zone]

#Save cleaned files
s2LAI_grouped.to_csv(import_path + '01_cleanedlaidataS2.csv')
s6LAI_grouped.to_csv(import_path + '01_cleanedlaidataS6.csv')

# %%

'''TIMESERIES BASICS'''
s2_grouped = s2data_df.groupby(by = ['zones', 'time']).mean().reset_index()
s6_grouped = s6data_df.groupby(by = ['zones', 'time']).mean().reset_index()

s2Bog = s2_grouped[s2_grouped.zones == 'Bog']
s2Lagg = s2_grouped[s2_grouped.zones == 'Lagg']
s2Upland = s2_grouped[s2_grouped.zones == 'Upland']

s6Bog = s6_grouped[s6_grouped.zones == 'Bog']
s6Lagg = s6_grouped[s6_grouped.zones == 'Lagg']
s6Upland = s6_grouped[s6_grouped.zones == 'Upland']

## Autocorrelationfunctions
#Is the variance, mean and autocovariance of the snow depth relatively stable over time?
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#S6
fig, ax = plt.subplots(3, 2, figsize = (12, 9))  

s6bog_acf = plot_acf(np.array(s6Bog.depths.squeeze()), ax = ax[0][0])
s6lagg_acf = plot_acf(np.array(s6Lagg.depths.squeeze()), ax = ax[1][0], title = ' ')
s6upland_acf = plot_acf(np.array(s6Upland.depths.squeeze()), ax = ax[2][0], title = ' ')
s6bog_pacf = plot_pacf(np.array(s6Bog.depths.squeeze()), lags = 6, ax = ax[0][1])
s6lagg_pacf = plot_pacf(np.array(s6Lagg.depths.squeeze()), lags = 6, ax = ax[1][1], title = ' ')
s6upland_pacf = plot_pacf(np.array(s6Upland.depths.squeeze()), lags = 6, ax = ax[2][1], title = ' ')

plt.suptitle('Autocorrelation functions in S6')
ax[0][0].set_xlabel('Bog')
ax[1][0].set_xlabel('Lagg')
ax[2][0].set_xlabel('Upland')
plt.show()

#S2
fig, ax = plt.subplots(3, 2, figsize = (12, 9))  

s2bog_acf = plot_acf(np.array(s2Bog.depths.squeeze()), ax = ax[0][0])
s2lagg_acf = plot_acf(np.array(s2Lagg.depths.squeeze()), ax = ax[1][0], title = ' ')
s2upland_acf = plot_acf(np.array(s2Upland.depths.squeeze()), ax = ax[2][0], title = ' ')
s2bog_pacf = plot_pacf(np.array(s2Bog.depths.squeeze()), lags = 6, ax = ax[0][1])
s2lagg_pacf = plot_pacf(np.array(s6Lagg.depths.squeeze()), lags = 6, ax = ax[1][1], title = ' ')
s2upland_pacf = plot_pacf(np.array(s2Upland.depths.squeeze()), lags = 6, ax = ax[2][1], title = ' ')

plt.suptitle('Autocorrelation functions in S2')
ax[0][0].set_xlabel('Bog')
ax[1][0].set_xlabel('Lagg')
ax[2][0].set_xlabel('Upland')
plt.show()

# %%

### Snow vs. Zone Repeated Measures
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols

#Remove NaN rows
s2_grouped = s2_grouped.dropna()

#Repeated Measures ANOVA
snowANOVA_s2 = ols('depths ~ C(zones, Sum)*C(time, Sum)', data = s2_grouped).fit()
table = sm.stats.anova_lm(snowANOVA_s2, typ=1, test = 'Chisq')
print(table)


# %%
