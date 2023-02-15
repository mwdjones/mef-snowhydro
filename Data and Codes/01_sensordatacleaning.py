# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:53:12 2022


@author: marie
"""

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

'''PROBE DATA IMPORT'''
directory = 'D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/Data and Codes/Raw Data/GroPoint/'
save_path = 'D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/Figures/soilPlots/'
all_files = glob.glob(os.path.join(directory, 'S*.txt'))

#For timestamp rounding 
dt_format = "%Y-%m-%d %H:%M"

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

def plotMoisture(df, save_path):
    fig, ax = plt.subplots(figsize=(6, 4),
                        layout="constrained")

    ax.plot(df.DateTime, df.SoilMoist_15cm, '-r',
     label = '15cm')
    ax.plot(df.DateTime, df.SoilMoist_30cm, '-b',
     label = '30cm')
    ax.plot(df.DateTime, df.SoilMoist_45cm, '-g',
     label = '45cm')

    plt.xlabel("Date")
    plt.ylabel("Soil Moisture")
    plt.title(str(df.SensorName[0]))

    plt.xlim(min(df.DateTime), max(df.DateTime))

    plt.legend()

    #plt.show()
    plt.savefig(save_path + "moistfig" + str(df.SensorName[0]) + ".pdf")
    plt.savefig(save_path + "moistfig" + str(df.SensorName[0]) + ".jpg")



def plotTemp(df, save_path):
    fig, ax = plt.subplots(figsize=(6, 4),
                    layout="constrained")

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

    plt.xlabel("Date")
    plt.ylabel("Soil Temperature")
    plt.title(str(df.SensorName[0]))

    plt.xlim(min(df.DateTime), max(df.DateTime))

    plt.legend()

    #plt.show()
    plt.savefig(save_path + "tempfig" + str(df.SensorName[0]) + ".pdf")
    plt.savefig(save_path + "tempfig" + str(df.SensorName[0]) + ".jpg")



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
    plotMoisture(df, save_path)

    #Plot soil temperature
    plotTemp(df, save_path)
    
    file_list.append(df)
    
#Concatenate all files in list
sensor_data = pd.concat(file_list, axis = 0, ignore_index = True)


'''EXPORT CSV FILE'''
sensor_data.to_csv(save_path + '01_cleanedsensordata.csv')
    



    





