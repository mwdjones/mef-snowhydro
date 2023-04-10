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

allSnow_df = pd.concat([s6data_df, s2data_df])

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
s6LAI_import = pd.read_csv(import_path_raw + "S6_winterLAI.txt", sep = ';')

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

def calc_corr(x, y, return_slope = False):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    
    if(return_slope):
        return slope, r_value**2
    else:
        return r_value**2
   

#%%
'''PLOTTING SPECIFICS'''
#Save Paths
save_path = "D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Figures/"

#Plotting Specifics
custom_pal = sns.color_palette(['#1b9e77', '#d95f02', '#7570b3'])

#%%
'''SNOW TIMESERIES AND KDE PLOTS'''

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
    palette = custom_pal, 
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
    #errorbar = 'ci', 
    palette = custom_pal, 
    ax = ax2, 
    legend = False
    )

ax2.set_xlim(min(s2data_df.time), max(s2data_df.time))
ax2.set_ylim(0, 70)
ax2.set_xlabel('Time')
ax2.set_ylabel('Snow depth [cm]')

plt.xticks(rotation=30)
plt.suptitle('Snow Depths in S2')
plt.savefig(save_path + 'snowPlots/' + 's2_kde_timeseries.pdf')
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
    palette = custom_pal, 
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
 #   errorbar = 'ci', 
    palette = custom_pal, 
    ax = ax2, 
    legend = False
    )

ax2.set_xlim(min(s6data_df.time), max(s6data_df.time))
ax2.set_ylim(0, 70)
ax2.set_xlabel('Time')
ax2.set_ylabel('Snow depth [cm]')

plt.xticks(rotation=30)
plt.suptitle('Snow Depths in S6')
plt.savefig(save_path + 'snowPlots/' + 's6_kde_timeseries.pdf')
plt.show()

#%%
'''SNOW DEPTH BY SLOPE AND ASPECT'''

#Plot the snow depths on polar coordinates with 0-360 aspect
#Dictate point size as slope - smaller slopes (bog) won't be as visible
fig, [ax1, ax2] = plt.subplots(1, 2, constrained_layout = True, 
                subplot_kw = {'projection':'polar'},
                figsize = (8,3))

plot1 = ax1.scatter(x = s2data_df["aspect"], y = s2data_df["depths"],
         c = s2data_df['slope'],
         s = s2data_df['slope'],
         vmin = 0, vmax = 25,
         zorder = 4)

plot2 = ax2.scatter(x = s6data_df["aspect"], y = s6data_df["depths"],
         c = s6data_df['slope'],
         s = s6data_df['slope'],
         vmin = 0, vmax = 25,
         zorder = 4)

fig.colorbar(plot2, ax = ax2, label = "Slope (%)")

ax1.set_title('S2')
ax1.set_theta_direction(-1)
ax1.set_theta_zero_location("N")
ax1.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

ax2.set_title('S6')
ax2.set_theta_direction(-1)
ax2.set_theta_zero_location("N")
ax2.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

plt.savefig(save_path + 'snowPlots/' + 'aspectPolarPlot.pdf')
plt.show()


#%%
'''LAI AND SNOW DEPTHS'''
# Plot summer LAI 
#Remove rows with NANs in the Zone column
s2data_df = s2data_df.dropna(subset = ['zones'])
s6data_df = s6data_df.dropna(subset = ['zones'])

#Declare dataframe
lai_corr = pd.DataFrame({'time' : [], 'zone' : [], 'corr' : [], 'slope' : [], 'watershed' : []})

#S2
#Loop through the available times
for t in set(s2data_df.time):
    #Loop through regions
    for z in ['Upland', 'Bog', 'Lagg']:
        #Subset the snow data to match that time
        snow_subset = s2data_df[(s2data_df.time == t) & (s2data_df.zones == z)]
        #Sort so stakes match
        snow_subset.sort_values('stakes')

        #Remove two cases for now -- add back in when Kristina sends LAI data
        if z == 'Upland':
            snow_subset = snow_subset[snow_subset["stakes"].str.contains("S214") == False]
  

        #Subset
        lai_subset = s2LAI_grouped[s2LAI_grouped.Zone == z]
        #Sort so stakes match
        lai_subset.sort_values('Stake_ID')
  
        #Calculate the correlation between the snowdepths and the LAI
        slope, r = calc_corr(snow_subset['depths'], lai_subset['LAI 4Ring'], return_slope = True)

        #Append to dataframe
        lai_corr = pd.concat([lai_corr, pd.DataFrame({'time': [t], 'zone': [z], 'corr': [r], 'slope': [slope], 'watershed': ['S2']})])

        #Loop through the available times

#S6       
for t in set(s6data_df.time):
    #Loop through regions
    for z in ['Upland', 'Bog', 'Lagg']:
        #Subset the snow data to match that time
        snow_subset = s6data_df[(s6data_df.time == t) & (s6data_df.zones == z)]
        #Sort so stakes match
        snow_subset.sort_values('stakes')

        #Remove two cases for now -- add back in when Kristina sends LAI data
        #if z == 'Upland':
        #    snow_subset = snow_subset[snow_subset["stakes"].str.contains("S601|S602") == False]
  

        #Subset
        lai_subset = s6LAI_grouped[s6LAI_grouped.Zone == z]
        #Sort so stakes match
        lai_subset.sort_values('Stake_ID')
  
        #Calculate the correlation between the snowdepths and the LAI
        slope, r = calc_corr(snow_subset['depths'], lai_subset['LAI 4Ring'], return_slope = True)

        #Append to dataframe
        lai_corr = pd.concat([lai_corr, pd.DataFrame({'time': [t], 'zone': [z], 'corr': [r], 'slope': [slope], 'watershed': ['S6']})])


#Plot
sns.set_style('white')
fig, [ax1, ax2] = plt.subplots(ncols=1, nrows=2, figsize=(5.5, 7),
                        layout="constrained")

#R^2 values
for t in set(lai_corr.time):
    ax1.vlines(pd.to_datetime(t), ymin = 0, ymax = 1, colors = 'silver', zorder = 1)

sns.scatterplot(x = pd.to_datetime(lai_corr['time']), y = lai_corr['corr'],
   hue = lai_corr['zone'], 
   style = lai_corr['watershed'],
   palette = custom_pal, 
   s = 80,
   linewidth = 0,
   ax = ax1)

plt.xticks(rotation=30)
#plt.legend(loc = 'upper right')

ax1.set_ylim(0, 1)
ax1.set_xlabel(' ')
ax1.set_ylabel(r'4Ring LAI and Snow Depth $R^2$')
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()

#Slope values
for t in set(lai_corr.time):
    ax2.vlines(pd.to_datetime(t), ymin = -0.5, ymax = 0.5, colors = 'silver', zorder = 1)

sns.scatterplot(x = pd.to_datetime(lai_corr['time']), y = lai_corr['slope'],
   hue = lai_corr['zone'], 
   style = lai_corr['watershed'],
   palette = custom_pal, 
   s = 80,
   linewidth = 0,
   ax = ax2)

plt.xticks(rotation=30)
#plt.legend(loc = 'upper right')

ax2.set_ylim(-0.2, 0.2)
ax2.set_xlabel(' ')
ax2.set_ylabel(r'4Ring LAI and Snow Depth $\beta$')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

plt.savefig(save_path + 'laiPlots/' + 'lai_SD_correlations.pdf')
plt.show()

#%%
'''LAI AND SNOW DEPTH - INDIVIDUAL PLOTS'''

#Plot sample individual corrplots
fig, [ax1, ax2, ax3] = plt.subplots(ncols=1, nrows=3, figsize=(3, 9),
                        layout="constrained"
                        )

#select data for time
snow_time_subset = s2data_df[s2data_df.time == '02-24-2023']

#quick patch
snow_time_subset.depths[11] = 13

#TIME = 2022-12-01, Bog
ax1_R = calc_corr(s2LAI_grouped[s2LAI_grouped.Zone == 'Bog']['LAI 4Ring'],
                  snow_time_subset[snow_time_subset.zones == 'Bog']['depths'])
sns.regplot(x = s2LAI_grouped[s2LAI_grouped.Zone == 'Bog']['LAI 4Ring'],
                 y = snow_time_subset[snow_time_subset.zones == 'Bog']['depths'], 
                 ax = ax1, 
                 color = custom_pal[1])
ax1.set_xlim(min(s2LAI_grouped[s2LAI_grouped.Zone == 'Bog']['LAI 4Ring']), 
             max(s2LAI_grouped[s2LAI_grouped.Zone == 'Bog']['LAI 4Ring']))
#ax1.set_ylim(0, 25)
ax1.set_ylabel('Snow depth [cm]')
#ax1.text(1.9, 22, r'$R^2$ = %.2f' % ax1_R)

#TIME = 2022-12-01, Upland
x = s2LAI_grouped[(s2LAI_grouped.Zone == 'Upland')]['LAI 4Ring']
y = snow_time_subset[(snow_time_subset.zones == 'Upland') & (snow_time_subset['stakes'].str.contains('S214') == False)]['depths']

ax2_R = calc_corr(x,y)
sns.regplot(x = x,
                 y = y, 
                 ax = ax2, 
                 color = custom_pal[0])
ax2.set_xlim(min(x), 
             max(x))
#ax2.set_ylim(0, 25)
ax2.set_ylabel('Snow depth [cm]')
#ax2.text(1.5, 22, r'$R^2$ = %.2f' % ax2_R)

#TIME = 2022-12-01, Lagg
ax3_R = calc_corr(s2LAI_grouped[s2LAI_grouped.Zone == 'Lagg']['LAI 4Ring'],
                  snow_time_subset[snow_time_subset.zones == 'Lagg']['depths'])
sns.regplot(x = s2LAI_grouped[s2LAI_grouped.Zone == 'Lagg']['LAI 4Ring'],
                 y = snow_time_subset[snow_time_subset.zones == 'Lagg']['depths'], 
                 ax = ax3, 
                 color = custom_pal[2])
ax3.set_xlim(min(s2LAI_grouped[s2LAI_grouped.Zone == 'Lagg']['LAI 4Ring']), 
             max(s2LAI_grouped[s2LAI_grouped.Zone == 'Lagg']['LAI 4Ring']))
#ax3.set_ylim(0, 25)
ax3.set_ylabel('Snow depth [cm]')
#ax3.text(1.5, 22, r'$R^2$ = %.2f' % ax3_R)

plt.savefig(save_path + 'laiPlots/' + 'sampCorrPlots.pdf')
plt.show()

#%%
'''FROST DEPTHS'''

#Plot Frost depths
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 3.5),
                        layout="constrained")

sns.boxplot(x = frost['DATE'].dt.date, y = frost['FROST.1'],
    hue = frost['Zones'], 
    palette = custom_pal, 
    boxprops={"edgecolor": "white",
                "linewidth": 0.5},
    ax = axs)

axs.set_xlabel(' ')
axs.set_ylabel('Frost Depth [cm]')

plt.savefig(save_path + 'frost_depths_timeseries.pdf')
plt.show()

#%%
'''SNOW AND FROST TIME SERIES'''

ylimit = 60
ylimit_frost = 25

def plotSnowSeries(gsx, gsy, data, frostData, color, xticks = False, labs = False):
    ax = plt.subplot(gs[gsx, gsy])

    #Precipitation
    Psnip = snipPrecip(precip, min(data.time), max(data.time))
    ax.bar(Psnip.Date, 30*Psnip.P_in, color = 'lightgrey', zorder = -2)

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
    if (labs):
        ax.set_ylabel('Snow Depth [cm]')
        ax2.set_ylabel('Frost Depth [cm]')
    else:
        ax.set_ylabel(' ')
        ax2.set_ylabel(' ')
    ax.set_xlim(min(data.time), max(data.time))
    ax.set_ylim(-ylimit, ylimit)

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
plotSnowSeries(1, 0, snowTemp_Lagg, frost_LaggFilled, '#7570b3', labs = True)
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
plotSnowSeries(1, 0, snowTemp_Lagg, frost_LaggFilled, '#7570b3', labs = True)
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

#ax1.set_xlim(min(soilData.DateTime), max(soilData.DateTime))
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

#ax2.set_xlim(min(soilData.DateTime), max(soilData.DateTime))
ax2.set_ylim(-3, 14)
ax2.set_ylabel('Soil Temperature at 5cm depth')
ax2.set_xlabel('Date')

# %%
