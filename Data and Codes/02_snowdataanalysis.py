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
import_path = "E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Cleaned Data/"
save_path = "E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Figures/snowPlots/"

#Import
s2data = xr.open_dataset(import_path + '01_cleanedsnowdataS2.nc')
s2data_df = s2data.to_dataframe().reset_index(drop = False)
s2data_df = s2data_df.replace('NaN', np.nan)
s2data_df = s2data_df.replace('nan', np.nan)
s2data_df['watershed'] = 'S2'

s6data = xr.open_dataset(import_path + '01_cleanedsnowdataS6.nc')
s6data_df = s6data.to_dataframe().reset_index(drop = False)
s6data_df = s6data_df.replace('NaN', np.nan)
s6data_df = s6data_df.replace('nan', np.nan)
s6data_df['watershed'] = 'S6'

data_all = pd.concat([s2data_df, s6data_df], ignore_index = True)

# %%

'''Plots'''
#Plot winter timeseries
#S2
fig = plt.figure(constrained_layout = True, 
    figsize = (9,3))

gs = GridSpec(1, 3, figure = fig)

#histogram
ax1 = fig.add_subplot(gs[0, 0])
sns.kdeplot(data = s2data_df, 
    x = 'depths', 
    hue = 'zones', 
    palette = 'rocket_r', 
    multiple = 'fill' 
    #fill = True
    )

ax1.set_xlabel('Snow depth [cm]')
ax1.set_ylim(0, 1)
sns.move_legend(ax1, "upper left")

#time series
ax2 = fig.add_subplot(gs[0, 1:3])
sns.lineplot(data = s2data_df, 
    x = 'time',
    y = 'depths',
    hue = 'zones', 
    errorbar = 'ci', 
    palette = 'rocket_r', 
    ax = ax2, 
    legend = False
    )

ax2.set_xlim('12-02-2022', '02-10-2023')
ax2.set_ylim(0, 50)
ax2.set_xlabel('Time')
ax2.set_ylabel('Snow depth [cm]')

plt.xticks(rotation=30)
plt.suptitle('Snow Depths in S2')
plt.savefig(save_path + 's2_kde_fill_timeseries.pdf')
plt.show()

#S6
#Plot winter timeseries
fig = plt.figure(constrained_layout = True, 
    figsize = (9,3))

gs = GridSpec(1, 3, figure = fig)

#histogram
ax1 = fig.add_subplot(gs[0, 0])
sns.kdeplot(data = s6data_df, 
    x = 'depths', 
    hue = 'zones', 
    palette = 'rocket_r', 
    multiple = 'fill'
    )

ax1.set_xlabel('Snow depth [cm]')
ax1.set_ylim(0, 1)
sns.move_legend(ax1, "upper left")

#time series
ax2 = fig.add_subplot(gs[0, 1:3])
sns.lineplot(data = s6data_df, 
    x = 'time',
    y = 'depths',
    hue = 'zones', 
    errorbar = 'ci', 
    palette = 'rocket_r', 
    ax = ax2, 
    legend = False
    )

ax2.set_xlim('12-02-2022', '02-10-2023')
ax2.set_ylim(0, 50)
ax2.set_xlabel('Time')
ax2.set_ylabel('Snow depth [cm]')

plt.xticks(rotation=30)
plt.suptitle('Snow Depths in S6')
plt.savefig(save_path + 's6_kde_fill_timeseries.pdf')
plt.show()


# %%
