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

s6data = xr.open_dataset(import_path + '01_cleanedsnowdataS6.nc')
s6data_df = s6data.to_dataframe().reset_index(drop = False)
s6data_df = s6data_df.replace('NaN', np.nan)
s6data_df = s6data_df.replace('nan', np.nan)
s6data_df.time = pd.to_datetime(s6data_df.time)
s6data_df['watershed'] = 'S6'


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

# %%


# %%
