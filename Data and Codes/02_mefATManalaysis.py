
#%%

'''PACKAGE IMPORTS'''
#For data analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr 

#For data import
import glob
import os

#For datetime 
import datetime
# %%

'''Import Raw Data'''
importRAW = "./Cleaned Data/ATM/"
figSave = "../Figures/ATM/"

NADPmet = pd.read_csv(importRAW + '01_cleanedNADP.csv', 
                      parse_dates = ['TIMESTAMP'])
Southmet = pd.read_csv(importRAW + '01_cleanedSouthMet.csv', 
                       parse_dates = ['TIMESTAMP'])
S2Forestmet = pd.read_csv(importRAW + '01_cleanedS2F.csv',
                        parse_dates = ['TIMESTAMP'])

# %%
#General Plotting Function
def plotMet(dat, var, HH = True):
    if(HH):
        window = 672
    else:
        window = 336

    #Plot data and rolling biweekly average
    fig, ax = plt.subplots(1, 1, figsize = (6,4))
    ax.plot(dat['TIMESTAMP'], dat[var], 
         linewidth = 1, color = 'red',
         alpha = 0.4)
    roll = dat[var].rolling(window).mean()
    ax.plot(dat['TIMESTAMP'], roll, linewidth = 1, color = 'red')

    #Figure Specifications
    ax.set_xlim(pd.to_datetime('09-01-2022'), pd.to_datetime('08-01-2023'))
    ax.set_ylabel(var)
    ax.set_xlabel('Timestamp')

    plt.show()


#%%
'''South Met plots'''
plotMet(dat = Southmet, var = 'AirT', HH = True)
plotMet(dat = Southmet, var = 'RH', HH = True)
plotMet(dat = Southmet, var = 'BP', HH = True)

'''NADP plots'''
plotMet(dat = NADPmet, var = 'Air_TempC')
plotMet(dat = NADPmet, var = 'RH')
plotMet(dat = NADPmet, var = 'WindSpd_S_WVT')
plotMet(dat = NADPmet, var = 'PAR_Den_Avg')

'''S2F plots'''
plotMet(dat = S2Forestmet, var = 'Air_TempC_Avg')
plotMet(dat = S2Forestmet, var = 'RH')
plotMet(dat = S2Forestmet, var = 'PAR_Den_Avg')
plotMet(dat = S2Forestmet, var = 'Soil_VWC_Avg')
plotMet(dat = S2Forestmet, var = 'WindSpd_S_WVT')

# %%
'''Under Canopy vs Open Canopy Comparison'''

#Wind Speed with weekly smoothing
fig, ax = plt.subplots(1, 1, figsize = (6,4))

#Forest Met Station
roll = S2Forestmet['WindSpd_S_WVT'].rolling(672).mean()
rollSD = S2Forestmet['WindSpd_S_WVT'].rolling(672).std()
ax.plot(S2Forestmet['TIMESTAMP'], roll,
        linewidth = 1, color = 'darkgreen', 
        label = 'Under Canopy')
ax.fill_between(S2Forestmet['TIMESTAMP'],
                roll - rollSD, roll + rollSD,
                color = 'darkgreen',
                alpha=0.2)

#Forest Met Station
roll2 = NADPmet['WindSpd_S_WVT'].rolling(672).mean()
roll2SD = NADPmet['WindSpd_S_WVT'].rolling(672).std()
ax.plot(NADPmet['TIMESTAMP'], roll2,
        linewidth = 1, color = 'lightgreen', 
        label = 'Open Canopy')
ax.fill_between(NADPmet['TIMESTAMP'],
                roll2 - roll2SD, roll2 + roll2SD,
                color = 'lightgreen',
                alpha=0.2)

#Figure Specifications
ax.set_xlim(pd.to_datetime('09-01-2022'), pd.to_datetime('08-01-2023'))
ax.set_ylim(0, 1.6)
ax.set_ylabel('Wind Speed [m/s]')
ax.set_xlabel('Timestamp')
ax.legend()
ax.set_title('Wind Speed Under Canopy vs. Open Canopy at in Marcell South Unit')
plt.xticks(rotation=30)
plt.savefig(figSave + "WindSpeed_CanopyNoCanopy.pdf", 
            bbox_inches = 'tight')

#PAR Density with weekly smoothing
fig, ax = plt.subplots(1, 1, figsize = (6,4))

#Forest Met Station
roll = S2Forestmet['PAR_Den_Avg'].rolling(672).mean()
rollSD = S2Forestmet['PAR_Den_Avg'].rolling(672).std()
ax.plot(S2Forestmet['TIMESTAMP'], roll,
        linewidth = 1, color = 'gold', 
        label = 'Under Canopy')
ax.fill_between(S2Forestmet['TIMESTAMP'],
                roll - rollSD, roll + rollSD,
                color = 'gold',
                alpha=0.2)

#Forest Met Station
roll2 = NADPmet['PAR_Den_Avg'].rolling(672).mean()
roll2SD = NADPmet['PAR_Den_Avg'].rolling(672).std()
ax.plot(NADPmet['TIMESTAMP'], roll2,
        linewidth = 1, color = 'tan', 
        label = 'Open Canopy')
ax.fill_between(NADPmet['TIMESTAMP'],
                roll2 - roll2SD, roll2 + roll2SD,
                color = 'tan',
                alpha=0.2)

#Figure Specifications
ax.set_xlim(pd.to_datetime('09-01-2022'), pd.to_datetime('08-01-2023'))
ax.set_ylim(0, 1200)
ax.set_ylabel('Plant Available Radiation')
ax.set_xlabel('Timestamp')
ax.legend()
ax.set_title('PAR Under Canopy vs. Open Canopy at in Marcell South Unit')
plt.xticks(rotation=30)
plt.savefig(figSave + "PAR_CanopyNoCanopy.pdf", 
            bbox_inches = 'tight')

#Air Temperature
fig, [ax, ax2] = plt.subplots(2, 1, 
                            figsize = (6,4), 
                            sharex = True)

#Forest Met Station
roll = S2Forestmet['Air_TempC_Avg'].rolling(672).mean()
rollSD = S2Forestmet['Air_TempC_Avg'].rolling(672).std()
ax.plot(S2Forestmet['TIMESTAMP'], roll,
        linewidth = 1, color = 'crimson', 
        label = 'Under Canopy')
ax.fill_between(S2Forestmet['TIMESTAMP'],
                roll - rollSD, roll + rollSD,
                color = 'crimson',
                alpha=0.2)

#Forest Met Station
roll2 = NADPmet['Air_TempC'].rolling(672).mean()
roll2SD = NADPmet['Air_TempC'].rolling(672).std()
ax.plot(NADPmet['TIMESTAMP'], roll2,
        linewidth = 1, color = 'salmon', 
        label = 'Open Canopy')
ax.fill_between(NADPmet['TIMESTAMP'],
                roll2 - roll2SD, roll2 + roll2SD,
                color = 'salmon',
                alpha=0.2)

#Difference plots
merge = pd.merge(NADPmet[['TIMESTAMP', 'Air_TempC']], S2Forestmet[['TIMESTAMP', 'Air_TempC_Avg']], on = 'TIMESTAMP')
diff = merge.Air_TempC - merge.Air_TempC_Avg
rollDiff = diff.rolling(672).mean()
rollDiffSD = diff.rolling(672).std()
ax2.plot(merge['TIMESTAMP'], rollDiff,
        linewidth = 1, color = 'grey')
ax2.fill_between(merge['TIMESTAMP'],
                rollDiff - rollDiffSD, rollDiff + rollDiffSD,
                color = 'grey',
                alpha=0.2)
ax2.hlines(y = 0, 
           xmin = pd.to_datetime('09-01-2022'),
           xmax = pd.to_datetime('08-01-2023'), 
           color = 'lightgrey')

#Figure Specifications
ax.set_xlim(pd.to_datetime('09-01-2022'), pd.to_datetime('08-01-2023'))
ax.set_ylabel('Air Temperature [C]')
ax2.set_ylabel('Difference [C]')
ax2.set_xlabel('Timestamp')
ax.legend()
ax.set_title('Air Temp Under Canopy vs. Open Canopy at in Marcell South Unit')
plt.xticks(rotation=30)
plt.savefig(figSave + "AirTemp_CanopyNoCanopy.pdf", 
            bbox_inches = 'tight')

# %%
'''Weekly Met Averages'''

jointMet = pd.merge(NADPmet[['TIMESTAMP', 'WindSpd_S_WVT', 'Air_TempC', 'PAR_Den_Avg']],
                    S2Forestmet[['TIMESTAMP', 'WindSpd_S_WVT', 'Air_TempC_Avg', 'PAR_Den_Avg']],
                    on = 'TIMESTAMP')
jointMet.columns = ['TIMESTAMP', 'NADPwind', 'NADPtempC', 'NADPPAR', 'S2Fwind', 'S2FtempC', 'S2FPAR']
weeklyMet = jointMet.groupby(pd.Grouper(key='TIMESTAMP', freq='W')).mean() 
weeklyMet = weeklyMet.reset_index()

# %%

weeklyMet.to_csv(importRAW + 'WeeklyATMSummary.csv')
# %%

'''Snow Accumulation, LAI and Wind Speed'''

#Import Snow Data
s2data = xr.open_dataset('./Cleaned Data/01_cleanedsnowdataS2.nc')
s2data_df = s2data.to_dataframe().reset_index(drop = False)
s2data_df = s2data_df.replace('NaN', np.nan)
s2data_df = s2data_df.replace('nan', np.nan)
s2data_df.time = pd.to_datetime(s2data_df.time)
s2data_df['watershed'] = 'S2'

s6data = xr.open_dataset('./Cleaned Data/01_cleanedsnowdataS6.nc')
s6data_df = s6data.to_dataframe().reset_index(drop = False)
s6data_df = s6data_df.replace('NaN', np.nan)
s6data_df = s6data_df.replace('nan', np.nan)
s6data_df.time = pd.to_datetime(s6data_df.time)
s6data_df['watershed'] = 'S6'

allSnow_df = pd.concat([s6data_df, s2data_df]).reset_index()

#Import Winter LAI Data
winterLAI = pd.read_csv('./Cleaned Data/calibratedWinterLAI.csv')

#Calculate the delta snow accumulation for the week
allSnow_df['deltaDepth'] = allSnow_df.groupby(["northing", "easting"])['depths'].diff().fillna(allSnow_df.depths)

#Merge LAI onto snow data
snowLAI_merged = pd.merge(allSnow_df[['time', 'depths', 'zones', 'stakes', 'watershed', 'deltaDepth']], 
                        winterLAI[['Stake_ID', 'OLS Prediction Ring 5', 'Summer LAI 5Ring']], 
                        left_on = 'stakes', right_on = 'Stake_ID')

# %%
fig, ax = plt.subplots(1, 1, constrained_layout = True, 
    figsize = (4,4))

sns.scatterplot(data = snowLAI_merged,
        x = 'OLS Prediction Ring 5', y = 'deltaDepth')

# %%
