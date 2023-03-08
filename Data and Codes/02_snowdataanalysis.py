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
import seaborn.objects as so

#%%

'''Import Data'''
import_path = "E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Cleaned Data/"
save_path = "E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Figures/snowPlots/"

#Import Snow Data
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


#Import LAI Data
'''DATA IMPORT'''
filepath = "E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Raw Data/"
save_path_lai = 'E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Figures/laiPlots/'
s2LAI_import = pd.read_csv(filepath + "S2_summerLAI.txt", sep = ';')
s6LAI_import = pd.read_csv(filepath + "S6_summerLAI.txt", sep = ';')

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
'''Plots'''

#%%
###Plot winter snow timeseries
###S2
fig = plt.figure(constrained_layout = True, 
    figsize = (9,3))

gs = GridSpec(1, 3, figure = fig)

#histogram
ax1 = fig.add_subplot(gs[0, 0])
sns.kdeplot(data = s2data_df, 
    x = 'depths', 
    hue = 'zones', 
    palette = 'rocket_r', 
    #multiple = 'fill' 
    fill = True
    )

ax1.set_xlabel('Snow depth [cm]')
#ax1.set_ylim(0, 1)
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

ax2.set_xlim('12-02-2022', '02-24-2023')
ax2.set_ylim(0, 70)
ax2.set_xlabel('Time')
ax2.set_ylabel('Snow depth [cm]')

plt.xticks(rotation=30)
plt.suptitle('Snow Depths in S2')
plt.savefig(save_path + 's2_kde_timeseries.pdf')
plt.show()

###S6
fig = plt.figure(constrained_layout = True, 
    figsize = (9,3))

gs = GridSpec(1, 3, figure = fig)

#histogram
ax1 = fig.add_subplot(gs[0, 0])
sns.kdeplot(data = s6data_df, 
    x = 'depths', 
    hue = 'zones', 
    palette = 'rocket_r', 
    #multiple = 'fill', 
    fill = True
    )

ax1.set_xlabel('Snow depth [cm]')
#ax1.set_ylim(0, 1)
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

ax2.set_xlim('12-02-2022', '02-24-2023')
ax2.set_ylim(0, 70)
ax2.set_xlabel('Time')
ax2.set_ylabel('Snow depth [cm]')

plt.xticks(rotation=30)
plt.suptitle('Snow Depths in S6')
plt.savefig(save_path + 's6_kde_timeseries.pdf')
plt.show()


# %%
# Plot summer LAI 

def calc_corr(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

#Remove rows with NANs in the Zone column
s2data_df = s2data_df.dropna(subset = ['zones'])
s6data_df = s6data_df.dropna(subset = ['zones'])

#Declare dataframe
lai_corr = pd.DataFrame({'time' : [], 'zone' : [], 'corr' : [], 'watershed' : []})

#Loop through the available times
for t in set(s2data_df.time):
    #Loop through regions
    for z in ['Upland', 'Bog', 'Lagg']:
        #Subset the snow data to match that time
        snow_subset = s2data_df[(s2data_df.time == t) & (s2data_df.zones == z)]
        #Sort so stakes match
        snow_subset.sort_values('stakes')

        #Subset
        lai_subset = s2LAI_grouped[s2LAI_grouped.Zone == z]
        #Sort so stakes match
        lai_subset.sort_values('Stake_ID')
  
        #Calculate the correlation between the snowdepths and the LAI
        r = calc_corr(snow_subset['depths'], lai_subset['LAI 4Ring'])

        #Append to dataframe
        lai_corr = pd.concat([lai_corr, pd.DataFrame({'time': [t], 'zone': [z], 'corr': [r], 'watershed': ['S2']})])

        #Loop through the available times

'''#Issue with the lengths -- need to double check that all the stakes are labelled the same in both data files
#S6       
for t in set(s6data_df.time):
    #Loop through regions
    for z in ['Upland', 'Bog', 'Lagg']:
        #Subset the snow data to match that time
        snow_subset = s6data_df[(s6data_df.time == t) & (s6data_df.zones == z)]
        #Sort so stakes match
        snow_subset.sort_values('stakes')

        #Subset
        lai_subset = s6LAI_grouped[s6LAI_grouped.Zone == z]
        #Sort so stakes match
        lai_subset.sort_values('Stake_ID')
  
        #Calculate the correlation between the snowdepths and the LAI
        r = calc_corr(snow_subset['depths'], lai_subset['LAI 4Ring'])
  
        #Append to dataframe
        lai_corr = pd.concat([lai_corr, pd.DataFrame({'time': [t], 'zone': [z], 'corr': [r], 'watersed': ['S6']})])
'''

#Plot
sns.set_style('white')
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 3.5),
                        layout="constrained")

for t in set(lai_corr.time):
    axs.vlines(pd.to_datetime(t), ymin = 0, ymax = 1, colors = 'silver', zorder = 1)

sns.scatterplot(x = pd.to_datetime(lai_corr['time']), y = lai_corr['corr'], hue = lai_corr['zone'], 
    palette = 'rocket_r', 
    ax = axs)

#for idx,row in lai_corr.iterrows():
#    x = pd.to_datetime(row[0]) #timestamp
#    y = row[2] #correlation
#    text = row[1] #zone
#    ax.text(x, y ,text, horizontalalignment = 'center')

plt.xticks(rotation=30)
plt.legend(loc = 'upper right')

axs.set_ylim(0, 1)
axs.set_xlabel('Time')
axs.set_ylabel(r'4Ring LAI and Snow Depth $R^2$')

plt.savefig(save_path_lai + 'lai_SD_correlations.pdf')
plt.show()

# %%
