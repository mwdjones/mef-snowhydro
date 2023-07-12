#%%
'''IMPORTS'''
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
import_path_raw = "D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Raw Data/"


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
#%%
'''Save Data'''
frost.to_csv('./Cleaned Data/2023frost_data.csv')