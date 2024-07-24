#%%
'''PACKAGE IMPORTS'''
#For data analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from matplotlib import offsetbox

#For data import
import glob
import os

#For datetime 
import datetime

#%%

'''IMPORT LAI DATA'''
filepath = "./Data and Codes/Raw Data/"

#S2
s2LAI_import = pd.read_csv(filepath + "S2_winterLAI.txt", sep = ';')

#S6
s6LAI_import = pd.read_csv(filepath + "S6_winterLAI.txt", sep = ';')

#Calibration Points
calibLAI_import = pd.read_csv(filepath + 'calibrationWinterLAI.txt', sep = ';')

### Cleaning
#Trim colnames
s2LAI_import.columns =[col.strip() for col in s2LAI_import.columns]
s6LAI_import.columns =[col.strip() for col in s6LAI_import.columns]
calibLAI_import.columns =[col.strip() for col in calibLAI_import.columns]

#Subset columns of interest
nameDict = {"User Field 1":"Stake", "User Field 2":"Orientation", "User Field 3":"Zone"}
s2LAI = s2LAI_import[["User Field 1", "User Field 2", "User Field 3", "% Sky Area", "% Mask Area", "% Cnpy Open", "% Site Open", "LAI 4Ring", "LAI 5Ring", "Date", "Time"]]
s2LAI = s2LAI.rename(columns = nameDict)
s6LAI = s6LAI_import[["User Field 1", "User Field 2", "User Field 3", "% Sky Area", "% Mask Area", "% Cnpy Open", "% Site Open", "LAI 4Ring", "LAI 5Ring", "Date", "Time"]]
s6LAI = s6LAI.rename(columns = nameDict)
calibLAI = calibLAI_import[["User Field 1", "User Field 2", "User Field 3", "% Sky Area", "% Mask Area", "% Cnpy Open", "% Site Open", "LAI 4Ring", "LAI 5Ring", "Date", "Time"]]
calibLAI = calibLAI.rename(columns = {"User Field 1":"Stake", "User Field 2":"Orientation", "User Field 3":"Glare"})

#Separate stake values into new column
s2LAI['Stake_ID'] = [name[0:4] for name in s2LAI.Stake]
s6LAI['Stake_ID'] = [name[0:4] for name in s6LAI.Stake]
calibLAI['Stake_ID'] = [name[0:4] for name in calibLAI.Stake]

###Groupby
#Group numerical values
s2LAI_groupednums = pd.DataFrame(s2LAI.groupby(["Stake_ID"], as_index = False).mean()) #Takes the mean of each variable, drops Date, Time, Orientation, and Zone because they are not ints
s6LAI_groupednums = pd.DataFrame(s6LAI.groupby(["Stake_ID"], as_index = False).mean()) #Takes the mean of each variable, drops Date, Time, Orientation, and Zone because they are not ints
calibLAI_groupednums = pd.DataFrame(calibLAI.groupby(["Stake_ID"], as_index = False).mean()) #Takes the mean of each variable, drops Date, Time, Orientation, and Zone because they are not ints

#Group string values
s2LAI_groupednames = pd.DataFrame(s2LAI.groupby(["Stake_ID"], as_index = False)['Zone'].max())
s6LAI_groupednames = pd.DataFrame(s6LAI.groupby(["Stake_ID"], as_index = False)['Zone'].max())
calibLAI_groupednames = pd.DataFrame(calibLAI.groupby(["Stake_ID"], as_index = False)['Glare'].max())

#Merge
s2LAI_grouped = s2LAI_groupednames.merge(s2LAI_groupednums, how = 'outer')
s6LAI_grouped = s6LAI_groupednames.merge(s6LAI_groupednums, how = 'outer')
calibLAI_grouped = calibLAI_groupednames.merge(calibLAI_groupednums, how = 'outer')

#Concat S2 and S6
s2LAI_grouped['Watershed'] = 'S2'
s6LAI_grouped['Watershed'] = 'S6'
ogData = pd.concat([s2LAI_grouped, s6LAI_grouped])

#Trim field names
calibLAI_grouped.Glare =[str(col.strip()) for col in calibLAI_grouped.Glare]
ogData.Zone =[str(col.strip()) for col in ogData.Zone]

#%%
'''DATA SETUP'''

#Merge data on calibration Stake IDs 
allData = calibLAI_grouped.merge(ogData[['Stake_ID', 'Zone', 'LAI 4Ring', 'LAI 5Ring']], on = 'Stake_ID', how = 'left')

#Rename
allData = allData.rename(columns = {"LAI 4Ring_x":"Calib LAI 4Ring", 
                                    "LAI 5Ring_x":"Calib LAI 5Ring", 
                                    "LAI 4Ring_y":"Original LAI 4Ring", 
                                    "LAI 5Ring_y":"Original LAI 5Ring"})


#%%
'''INITIAL DATA EXAMINATION'''

fig, [ax1, ax2] = plt.subplots(1, 2, figsize = (6, 3.5), 
                               layout = 'constrained', 
                               sharey = True)

sns.scatterplot(allData['Original LAI 4Ring'], allData['Calib LAI 4Ring'], 
            hue = allData['Zone'], 
            ax = ax1)
ax1.axline([0,0], [3,3], color = 'r')
ax1.set_aspect('equal')

ax1.set_ylim(0,3)
ax1.set_xlim(0,3)
ax1.set_xlabel('Original LAI Data')
ax1.set_ylabel('Calibration LAI Data')
ax1.set_title("Ring 4")

sns.scatterplot(allData['Original LAI 5Ring'], allData['Calib LAI 5Ring'], 
                hue = allData['Zone'], 
                ax = ax2)
ax2.axline([0,0], [3,3], color = 'r')
ax2.set_aspect('equal')

ax2.set_ylim(0,3)
ax2.set_xlim(0,3)
ax2.set_xlabel('Original LAI Data')
ax2.set_title("Ring 5")

#%%
'''CALIBRATION'''
weights = np.where(allData.Glare == 'No Glare', 2, 1)
x = sm.add_constant(allData['Original LAI 4Ring'])
x2 = sm.add_constant(allData['Original LAI 5Ring'])
y = allData['Calib LAI 4Ring']
y2 = allData['Calib LAI 5Ring']

#Normal regression - no weights
normModel = sm.OLS(y, x).fit()
y_norm = normModel.predict(x)

normModel5 = sm.OLS(y2, x2).fit()
y_norm5 = normModel5.predict(x2)

#Weighted regression - higher weight to photos with no glare
weightModel = sm.WLS(y, x, weights).fit()
y_weight = weightModel.predict(x)

weightModel5 = sm.WLS(y2, x2, weights).fit()
y_weight5 = weightModel5.predict(x2)

# RING 4
fig, ax1 = plt.subplots(1, 1, figsize = (3.5, 3.5), 
                               layout = 'constrained')

#Scatterplot Data
sns.scatterplot(x['Original LAI 4Ring'], y, 
            size = weights,
            sizes = [30,60], 
            ax = ax1, 
            legend = None)

#Normal regression line
plt.plot(x['Original LAI 4Ring'], y_norm, c = 'red', label = 'OLS')

#Weighted Regression line
plt.plot(x['Original LAI 4Ring'], y_weight, c = 'blue', label = 'WLS')

#plot specifics
fig.legend(loc = 'upper left')
ax1.set_aspect('equal')

plt.show()

#Plot RING 5
fig, ax = plt.subplots(1, 1, figsize = (3.5, 3.5), 
                               layout = 'constrained')

#Scatterplot Data
sns.scatterplot(x2['Original LAI 5Ring'], y2, 
            size = weights,
            sizes = [30,60], 
            ax = ax, 
            legend = None)

#Normal regression line
plt.plot(x2['Original LAI 5Ring'], y_norm5, c = 'red', label = 'OLS')

#Weighted Regression line
plt.plot(x2['Original LAI 5Ring'], y_weight5, c = 'blue', label = 'WLS')

#plot specifics
fig.legend(loc = 'upper left')
ax.set_aspect('equal')

plt.show()

#Predict all og data using models
ogData['OLS Prediction Ring 4'] = normModel.predict(sm.add_constant(ogData['LAI 4Ring']))
ogData['WLS Prediction Ring 4'] = weightModel.predict(sm.add_constant(ogData['LAI 4Ring']))

ogData['OLS Prediction Ring 5'] = normModel5.predict(sm.add_constant(ogData['LAI 5Ring']))
ogData['WLS Prediction Ring 5'] = weightModel5.predict(sm.add_constant(ogData['LAI 5Ring']))

#%%
'''CHECK CALIBRATION'''

#Transformed Winter LAI boxplot
fig, [ax1, ax2] = plt.subplots(1, 2, figsize = (5.5, 3.5), 
                               layout = 'constrained')

#OLS
sns.boxplot(data = ogData, x = 'Zone', y = 'OLS Prediction Ring 4',
    notch=False, showcaps=False,
    hue = 'Watershed',
    flierprops={"marker": "x"},
    #boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax1)
ax1.set_ylabel('LAI Ring 4')
at = offsetbox.AnchoredText(
    "OLS", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)

#WLS
sns.boxplot(data = ogData, x = 'Zone', y = 'WLS Prediction Ring 4',
    notch=False, showcaps=False,
    hue = 'Watershed',
    flierprops={"marker": "x"},
    #boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax2)
ax2.set_ylabel('LAI Ring 4')
at = offsetbox.AnchoredText(
    "WLS", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)

plt.show()

#Transformed Winter LAI boxplot
fig, [ax1, ax2] = plt.subplots(1, 2, figsize = (5.5, 3.5), 
                               layout = 'constrained')

#OLS
sns.boxplot(data = ogData, x = 'Zone', y = 'OLS Prediction Ring 5',
    notch=False, showcaps=False,
    hue = 'Watershed',
    flierprops={"marker": "x"},
    #boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax1)
ax1.set_ylabel('LAI Ring 4')
at = offsetbox.AnchoredText(
    "OLS", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)

#WLS
sns.boxplot(data = ogData, x = 'Zone', y = 'WLS Prediction Ring 5',
    notch=False, showcaps=False,
    hue = 'Watershed',
    flierprops={"marker": "x"},
    #boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax2)
ax2.set_ylabel('LAI Ring 4')
at = offsetbox.AnchoredText(
    "WLS", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)

plt.show()

# %%
'''CHECK AGAINST SUMMER DATA'''
#Import
s2LAI_import = pd.read_csv(filepath + "S2_summerLAI.txt", sep = ';')
s6LAI_import = pd.read_csv(filepath + "S6_summerLAI.txt", sep = ';')

### Cleaning
#Trim colnames
s2LAI_import.columns =[col.strip() for col in s2LAI_import.columns]
s6LAI_import.columns =[col.strip() for col in s6LAI_import.columns]

#Subset columns of interest
nameDict = {"User Field 1":"Stake", "User Field 2":"Orientation", "User Field 3":"Zone"}
s2LAI = s2LAI_import[["User Field 1", "User Field 2", "User Field 3", "% Sky Area", "% Mask Area", "% Cnpy Open", "% Site Open", "LAI 4Ring", "LAI 5Ring", "Date", "Time"]]
s2LAI = s2LAI.rename(columns = nameDict)
s6LAI = s6LAI_import[["User Field 1", "User Field 2", "User Field 3", "% Sky Area", "% Mask Area", "% Cnpy Open", "% Site Open", "LAI 4Ring", "LAI 5Ring", "Date", "Time"]]
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

#Concat S2 and S6
s2LAI_grouped['Watershed'] = 'S2'
s6LAI_grouped['Watershed'] = 'S6'
ogData_summer = pd.concat([s2LAI_grouped, s6LAI_grouped])


#Trim field names
ogData_summer.Zone =[str(col.strip()) for col in ogData_summer.Zone]

#Merge column
ogData = ogData.merge(ogData_summer[["Stake_ID", "LAI 4Ring", "LAI 5Ring"]], 
                      on = 'Stake_ID')
ogData = ogData.rename(columns = {"LAI 4Ring_y":"Summer LAI 4Ring", 
                                  "LAI 4Ring_x":"LAI 4Ring", 
                                  "LAI 5Ring_y":"Summer LAI 5Ring", 
                                  "LAI 5Ring_x":"LAI 5Ring"})

#%%
### Plot - New Data RING 4
g = sns.FacetGrid(data = ogData,
            col = 'Watershed', 
            hue = 'Zone', 
            height = 3, aspect = 1, 
            legend_out = True)

g.map(sns.scatterplot, 'Summer LAI 4Ring', 'OLS Prediction Ring 4')
g.add_legend()

ax1, ax2 = g.axes[0]
ax1.axline([0,0], [2.75, 2.75], c = 'red', zorder = -1)
ax2.axline([0,0], [2.75, 2.75], c = 'red', zorder = -1)

plt.show()

### Plot - New Data RING 5
g = sns.FacetGrid(data = ogData,
            col = 'Watershed', 
            hue = 'Zone', 
            height = 3, aspect = 1, 
            legend_out = True)

g.map(sns.scatterplot, 'Summer LAI 5Ring', 'OLS Prediction Ring 5')
g.add_legend()

ax1, ax2 = g.axes[0]
ax1.axline([0,0], [2.75, 2.75], c = 'red', zorder = -1)
ax2.axline([0,0], [2.75, 2.75], c = 'red', zorder = -1)

plt.show()

### Plot - Old Data for reference
g = sns.FacetGrid(data = ogData,
            col = 'Watershed', 
            hue = 'Zone', 
            height = 3, aspect = 1, 
            legend_out = True)

g.map(sns.scatterplot, 'Summer LAI 4Ring', 'LAI 4Ring')
g.add_legend()

ax1, ax2 = g.axes[0]
ax1.axline([0,0], [2.75, 2.75], c = 'red', zorder = -1)
ax2.axline([0,0], [2.75, 2.75], c = 'red', zorder = -1)

plt.show()

### Plot - Old Data for reference
g = sns.FacetGrid(data = ogData,
            col = 'Watershed', 
            hue = 'Zone', 
            height = 3, aspect = 1, 
            legend_out = True)

g.map(sns.scatterplot, 'Summer LAI 5Ring', 'LAI 5Ring')
g.add_legend()

ax1, ax2 = g.axes[0]
ax1.axline([0,0], [2.75, 2.75], c = 'red', zorder = -1)
ax2.axline([0,0], [2.75, 2.75], c = 'red', zorder = -1)

plt.show()

# %%

#Export data frame to CSV
ogData.to_csv('D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Cleaned Data/calibratedWinterLAI.csv')

#Export Data separated by watershed
ogData[ogData.Watershed == 'S2'].to_csv('D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Cleaned Data/S2_winterLAI_calibrated.csv')
ogData[ogData.Watershed == 'S6'].to_csv('D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Cleaned Data/S6_winterLAI_calibrated.csv')


# %%
