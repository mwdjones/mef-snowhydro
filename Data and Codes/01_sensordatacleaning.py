# -*- coding: utf-8 -*-

"""
Created on Tue Oct 25 13:53:12 2022


@author: marie
"""

#%%

'''PACKAGE IMPORTS'''
#For data analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#For data import
import glob
import os

#For datetime 
import datetime

#%%

'''PROBE DATA IMPORT'''
#Soil Data from Gropoint sensors
directory = 'D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Raw Data/GroPoint/'
save_path = 'D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Figures/soilPlots/'
import_path_frost = "D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Raw Data/"
all_files = glob.glob(os.path.join(directory, 'S*.txt'))
bog_files = glob.glob(os.path.join(directory, 'S*.csv'))

#Precipitation data - update from MN DNR site occasionally, eventually replace with MEF data
precip_directory = 'D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Raw Data/'
precip = pd.read_csv(precip_directory + 'GrandRapids_Precip_MNDNR.csv', 
                     na_values = ['T', 'M'], 
                     parse_dates = ['Date'], 
                     names = ['Date', 'Tmax_F', 'Tmin_F', 'P_in', 'Snow_in', 'SnowDepth_in'], 
                     header = 0, 
                     dtype = {'P_in':float, 'Snow_in':float, 'SnowDepth_in':float})

#Frost Data from Anne
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
#For timestamp rounding 
dt_format = "%Y-%m-%d %H:%M"

#%%
'''FUNCTIONS'''

def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)

def plotMoisture(df, P, save_path):
    fig, ax = plt.subplots(figsize=(6, 4),
                        layout="constrained")
    
    
    #Soil Moisture
    ax.plot(df.DateTime, df.SoilMoist_15cm, '-r',
     label = '15cm')
    ax.plot(df.DateTime, df.SoilMoist_30cm, '-b',
     label = '30cm')
    ax.plot(df.DateTime, df.SoilMoist_45cm, '-g',
     label = '45cm')
    
    ax.set_ylabel("Soil Moisture")

    #Precip
    ax2 = ax.twinx()
    Psnip = snipPrecip(P, min(df.DateTime), max(df.DateTime))
    ax2.bar(Psnip.Date, Psnip.P_in, color = 'lightgrey', zorder = -2)

    ax2.set_ylabel("Precipitation [in]")

    plt.xlabel("Date")
    plt.title(str(df.SensorName[0]))

    plt.xlim(min(df.DateTime), max(df.DateTime))

    ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False)

    ax.legend()

    #plt.show()
    plt.savefig(save_path + "moistfig" + str(df.SensorName[0]) + ".pdf")
    plt.savefig(save_path + "moistfig" + str(df.SensorName[0]) + ".jpg")

    plt.show()

def plotTemp(df, P, save_path):
    fig, ax = plt.subplots(figsize=(6, 4),
                    layout="constrained")
    
    #Precip
    ax2 = ax.twinx()
    Psnip = snipPrecip(P, min(df.DateTime), max(df.DateTime))
    ax2.bar(Psnip.Date, Psnip.P_in, color = 'lightgrey')

    ax2.set_ylabel("Precipitation [in]")

    #Soil Temperature
    ax.plot(df.DateTime, df.SoilTemp1,
     label = '5cm')
    ax.plot(df.DateTime, df.SoilTemp2,
     label = '15cm')
    ax.plot(df.DateTime, df.SoilTemp3, 
     label = '25cm')
    ax.plot(df.DateTime, df.SoilTemp4, 
     label = '35cm')
    ax.plot(df.DateTime, df.SoilTemp5, 
     label = '45cm')
    ax.plot(df.DateTime, df.SoilTemp6, 
     label = '55cm')
    
    ax.set_ylabel("Soil Temperature")

    plt.xlabel("Date")
    plt.title(str(df.SensorName[0]))

    plt.xlim(min(df.DateTime), max(df.DateTime))

    ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False)

    ax.legend()

    #plt.show()
    plt.savefig(save_path + "tempfig" + str(df.SensorName[0]) + ".pdf")
    plt.savefig(save_path + "tempfig" + str(df.SensorName[0]) + ".jpg")

    plt.show()

def plotTemp_Heatmap(df, save_path):
    #Aggregate to daily - makes contour smoother
    df['DateTime'] = df['DateTime'].astype('datetime64[ns]')
    df2 = df.resample('D', on = 'DateTime').mean().reset_index()

    #select soil temp data
    data = df2.filter(regex = r'SoilTemp')

    #dates 
    dates = df2.DateTime

    #depths 
    depths = np.array([5, 15, 25, 35, 45, 55]) #depths in centimeters

    #levels
    levels = np.arange(-2, 14, 2)

    #plot heatmap
    fig, ax = plt.subplots(figsize = (8, 4),
                           layout = 'constrained')
    
    cset1 = ax.contourf(dates, depths, data.T, 
                    levels = levels,
                    cmap = 'coolwarm')
    cset2 = ax.contour(dates, depths, data.T, 
                    levels = cset1.levels,
                    colors = 'black')
    
    #frost contour line
    cset3 = ax.contour(dates, depths, data.T, 
                       levels = (0,), 
                       colors = 'red', 
                       linewidth = 2)
    
    #Add manual data
    #Match sensor name
    frostTemp = all_frost[(all_frost['STAKE NO'].str.contains(df.SensorName[0][-2:])) & (all_frost['FROST.1'] >= 5)]
 
    #Plots
    ax.scatter(frostTemp['DATE'], frostTemp['FROST.1'], 
            marker = 'X', 
            edgecolors = 'black', 
            facecolor = 'white')
    
    ax.invert_yaxis()
    fig.colorbar(cset1, ax = ax, label = 'Soil Temperature [degC]')
    plt.title(str(df.SensorName[0]))


    plt.savefig(save_path + "tempfig_heatmap" + str(df.SensorName[0]) + ".pdf")
    plt.savefig(save_path + "tempfig" + str(df.SensorName[0]) + ".jpg")

    plt.show()

def plotMoisture_Heatmap(df, save_path):
    #Aggregate to daily - makes contour smoother
    df['DateTime'] = df['DateTime'].astype('datetime64[ns]')
    df2 = df.resample('D', on = 'DateTime').mean().reset_index()

    #select soil temp data
    data = df2.filter(regex = r'SoilMoist')

    #dates 
    dates = df2.DateTime

    #depths 
    depths = np.array([15, 30, 45]) #depths in centimeters

    #levels
    levels = np.arange(0, 40, 5)

    #plot heatmap
    fig, ax = plt.subplots(figsize = (8, 4),
                           layout = 'constrained')
    
    cset1 = ax.contourf(dates, depths, data.T, 
                    levels = levels,
                    cmap = 'viridis_r')
    cset2 = ax.contour(dates, depths, data.T, 
                    levels = cset1.levels,
                    colors = 'black')
    
    ax.invert_yaxis()
    fig.colorbar(cset1, ax = ax, label = 'Soil Moisture [cm3/cm3]')
    plt.title(str(df.SensorName[0]))


    plt.savefig(save_path + "moistfig_heatmap" + str(df.SensorName[0]) + ".pdf")
    plt.savefig(save_path + "moistfig" + str(df.SensorName[0]) + ".jpg")

    plt.show()


def snipPrecip(P, fir, la):
    #function trims the precip log to the specified dates, filling in NaNs where there is no available data
    #P must be a timeseries containing at least two columns, one with dates and one with precip values
    range = pd.date_range(start = fir.date(), end = la.date())
    return P.set_index('Date').reindex(range).rename_axis('Date').reset_index()
   


#%%
'''Plots'''

file_list = []
for file in all_files:
    df = pd.read_csv(file, 
                     sep = ',',
                     header = None, 
                     names = ['DateTime', 'SensorAddress',
                              'SoilMoist_15cm', 'SoilMoist_30cm', 'SoilMoist_45cm', 
                              'SoilTemp1', 'SoilTemp2', 'SoilTemp3', 'SoilTemp4', 'SoilTemp5', 'SoilTemp6'])
    
    df['SensorName'] = file[file.find('\S')+1:file.find('\S')+5]
    
    for i in range(0, len(df['DateTime'])):
        #Converts to datetime format
        df.DateTime[i] = datetime.datetime.strptime(df.DateTime[i], dt_format)
        #Rounds to the nearest hour
        df.DateTime[i] = roundTime(df.DateTime[i], roundTo = 60*60)
    
    #Plot soil moisture
    plotMoisture(df, precip, save_path)

    #Plot soil temperature
    plotTemp(df, precip, save_path)

    #plot soil temp heatmap
    plotTemp_Heatmap(df, save_path)

    #plot soil moist heatmap
    plotMoisture_Heatmap(df, save_path)
    
    file_list.append(df)
    
#Concatenate all files in list
sensor_data = pd.concat(file_list, axis = 0, ignore_index = True)

#%%
### Bog sensors
bog_file_list = []
for file in bog_files:
    df = pd.read_csv(file, 
                     sep = ',', 
                     header = 1, 
                     names = ['DateTime', 'WaterTemp_C',
                              'Host Connected', 'End of File'], 
                     parse_dates = ['DateTime'])
    
    df['SensorName'] = file[file.find('\S')+1:file.find('\S')+5]
    df['SensorDepth_cm'] = file[file.find('\S')+6:file.find('\S')+8]
    
    #for i in range(0, len(df['DateTime'])):
    #    #Converts to datetime format
    #   df.DateTime[i] = datetime.datetime.strptime(df.DateTime[i], dt_format)
    #    #Rounds to the nearest hour
    #    df.DateTime[i] = roundTime(df.DateTime[i], roundTo = 60*60)

    bog_file_list.append(df)

#Concatenate all files in list
bog_sensor_data = pd.concat(bog_file_list, axis = 0, ignore_index = True)

#Plot bog data
for site in set(bog_sensor_data.SensorName):
    data = bog_sensor_data[bog_sensor_data.SensorName == site].reset_index(drop = True)

    fig, ax = plt.subplots(figsize=(6, 4),
                    layout="constrained")
    
    #Precip
    #ax2 = ax.twinx()
    #Psnip = snipPrecip(precip, min(df.DateTime), max(df.DateTime))
    #ax2.bar(Psnip.Date, Psnip.P_in, color = 'lightgrey')

    #ax2.set_ylabel("Precipitation [in]")

    #Soil Temperature
    for depth in set(data.SensorDepth_cm):
        plt.plot(data[data.SensorDepth_cm == depth].DateTime, data[data.SensorDepth_cm == depth].WaterTemp_C, label = str(depth) + 'cm')
    
    ax.set_ylabel("Water Temperature [C]")

    plt.xlabel("Date")
    plt.title(str(data.SensorName[0]))

    plt.xlim(min(data.DateTime), max(data.DateTime))

    #ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False)

    ax.legend()

    #plt.show()
    plt.savefig(save_path + "tempfig" + str(data.SensorName[0]) + "_" + str(df.SensorDepth_cm[0]) + ".pdf")
    plt.savefig(save_path + "tempfig" + str(data.SensorName[0]) + "_" + str(df.SensorDepth_cm[0]) + ".jpg")

    plt.show()
#%%

'''EXPORT CSV FILE'''
sensor_data.to_csv(save_path + '01_cleanedsensordata.csv')
    



    






# %%
