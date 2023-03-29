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
directory = 'E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Raw Data/GroPoint/'
save_path = 'E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Figures/soilPlots/'
all_files = glob.glob(os.path.join(directory, 'S*.txt'))

#Precipitation data - update from MN DNR site occasionally, eventually replace with MEF data
precip_directory = 'E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Raw Data/'
precip = pd.read_csv(precip_directory + 'GrandRapids_Precip_MNDNR.csv', 
                     na_values = ['T', 'M'], 
                     parse_dates = ['Date'], 
                     names = ['Date', 'Tmax_F', 'Tmin_F', 'P_in', 'Snow_in', 'SnowDepth_in'], 
                     header = 0, 
                     dtype = {'P_in':float, 'Snow_in':float, 'SnowDepth_in':float})

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
    
    file_list.append(df)
    
#Concatenate all files in list
sensor_data = pd.concat(file_list, axis = 0, ignore_index = True)

#%%

'''EXPORT CSV FILE'''
sensor_data.to_csv(save_path + '01_cleanedsensordata.csv')
    



    






# %%
