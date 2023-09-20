# -*- coding: utf-8 -*-
#%%
"""
Created on Tue Oct 25 13:53:12 2022


@author: marie
"""

'''PACKAGE IMPORTS'''
#For data analysis
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import seaborn as sns
import scipy

#%%

'''DATA IMPORT'''
filepath = "./Raw Data/"
filepath_winter = "./Cleaned Data/"
save_path = '../Figures/laiPlots/'
s2LAI_import = pd.read_csv(filepath + "S2_summerLAI.txt", sep = ';')
s6LAI_import = pd.read_csv(filepath + "S6_summerLAI.txt", sep = ';')

s2LAI_winter_grouped = pd.read_csv(filepath_winter + "S2_winterLAI_calibrated.csv")
s6LAI_winter_grouped = pd.read_csv(filepath_winter + "S6_winterLAI_calibrated.csv")

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

#Trim Zone Names
s2LAI_grouped.Zone =[zone.strip() for zone in s2LAI_grouped.Zone]
s6LAI_grouped.Zone =[zone.strip() for zone in s6LAI_grouped.Zone]

#Save cleaned files
s2LAI_grouped.to_csv(filepath_winter + '01_cleanedlaidataS2.csv')
s6LAI_grouped.to_csv(filepath_winter + '01_cleanedlaidataS6.csv')

#%%
'''PLOTS ANALYSIS - SUMMER'''

###LAI4 and LAI5 comparison by stake
fig, [ax1, ax2] = plt.subplots(ncols=1, nrows=2, figsize=(3.5, 5.5),
                        layout="constrained")
#S2
ax1.plot(s2LAI_grouped['Stake_ID'], s2LAI_grouped['LAI 4Ring'])
ax1.plot(s2LAI_grouped['Stake_ID'], s2LAI_grouped['LAI 5Ring'], linestyle = '--')
ax1.tick_params(axis = 'x', rotation = 90)
at = offsetbox.AnchoredText(
    "S2", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)
ax1.set_ylabel('LAI')
#S6
ax2.plot(s6LAI_grouped['Stake_ID'], s6LAI_grouped['LAI 4Ring'])
ax2.plot(s6LAI_grouped['Stake_ID'], s6LAI_grouped['LAI 5Ring'], linestyle = '--')
ax2.tick_params(axis = 'x', rotation = 90)
at = offsetbox.AnchoredText(
    "S6", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)
ax2.set_ylabel('LAI')

plt.suptitle("LAI Ring 4 and 5 data in S2 and S6")
plt.savefig(save_path + "LAI_ring4_5_comp.pdf")

###Boxplot of LAI4 values by zone
fig, [ax1, ax2] = plt.subplots(ncols=1, nrows=2, figsize=(3.5, 5.5),
                        layout="constrained")

#S2
sns.boxplot(data = s2LAI_grouped, x = 'Zone', y = 'LAI 4Ring',
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax1)
ax1.set_ylabel('LAI Ring 4')
at = offsetbox.AnchoredText(
    "S2", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)

#S6
sns.boxplot(data = s6LAI_grouped, x = 'Zone', y = 'LAI 4Ring',
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax2)
ax2.set_ylabel('LAI Ring 4')
at = offsetbox.AnchoredText(
    "S6", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)

plt.suptitle("LAI Ring 4 by peatland zone")
plt.savefig(save_path + "LAI_ring4_peatZone.pdf")


###Boxplot of LAI5 values by zone
fig, [ax1, ax2] = plt.subplots(ncols=1, nrows=2, figsize=(3.5, 5.5),
                        layout="constrained")

#S2
sns.boxplot(data = s2LAI_grouped, x = 'Zone', y = 'LAI 5Ring',
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax1)
ax1.set_ylabel('LAI Ring 4')
at = offsetbox.AnchoredText(
    "S2", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)

#S6
sns.boxplot(data = s6LAI_grouped, x = 'Zone', y = 'LAI 5Ring',
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax2)
ax2.set_ylabel('LAI Ring 4')
at = offsetbox.AnchoredText(
    "S6", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)

plt.suptitle("LAI Ring 5 by peatland zone")
plt.savefig(save_path + "LAI_ring5_peatZone.pdf")

###Boxplot of LAI by peatland
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 3.5),
                        layout="constrained")

sns.boxplot(data = pd.concat([s2LAI_grouped, s6LAI_grouped], keys = ['S2', 'S6']).reset_index(),
    x = 'level_0', y = 'LAI 4Ring',
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax)
ax.set_ylabel('LAI Ring 4')
ax.set_xlabel('Watershed')
at = offsetbox.AnchoredText(
    "S2", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)

plt.suptitle("LAI Ring 4 by Peatland")
plt.savefig(save_path + "LAI_ring4_peat.pdf")

#%%
'''PLOT ANALYSIS - WINTER'''

###LAI4 and LAI5 comparison by stake
fig, [ax1, ax2] = plt.subplots(ncols=1, nrows=2, figsize=(5, 3),
                        layout="constrained")
#S2
ax1.plot(s2LAI_winter_grouped['Stake_ID'], s2LAI_winter_grouped['OLS Prediction Ring 4'])
ax1.plot(s2LAI_winter_grouped['Stake_ID'], s2LAI_winter_grouped['OLS Prediction Ring 5'], linestyle = '--')
ax1.tick_params(axis = 'x', rotation = 90)
at = offsetbox.AnchoredText(
    "S2", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)
ax1.set_ylabel('LAI')

#S6
ax2.plot(s6LAI_winter_grouped['Stake_ID'], s6LAI_winter_grouped['OLS Prediction Ring 4'])
ax2.plot(s6LAI_winter_grouped['Stake_ID'], s6LAI_winter_grouped['OLS Prediction Ring 5'], linestyle = '--')
ax2.tick_params(axis = 'x', rotation = 90)
at = offsetbox.AnchoredText(
    "S6", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)
ax2.set_ylabel('LAI')


plt.suptitle("LAI Ring 4 and 5 data in S2 and S6, Winter")
plt.savefig(save_path + "LAI_winter_ring4_5_comp.pdf")

###Boxplot of LAI4 values by zone
fig, [ax1, ax2] = plt.subplots(ncols=2, nrows=1, figsize=(7, 3),
                        layout="constrained",
                        sharey = True)

#S2 LAI4
sns.boxplot(data = s2LAI_winter_grouped, x = 'Zone', y = 'OLS Prediction Ring 5',
    notch=False, showcaps=False,
    palette = sns.color_palette(['#1b9e77', '#d95f02', '#7570b3']),
    flierprops={"marker": "x"},
    ax = ax1)

ax1.set_ylabel('LAI Ring 5')
at = offsetbox.AnchoredText(
    "S2", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)

#S6 LAI4
sns.boxplot(data = s6LAI_winter_grouped, x = 'Zone', y = 'OLS Prediction Ring 5',
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    palette = sns.color_palette(['#1b9e77', '#d95f02', '#7570b3']),
    ax = ax2)

ax2.set_ylabel('LAI Ring 5')
at = offsetbox.AnchoredText(
    "S6", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)

#plt.suptitle("LAI Ring 4 in S2,S6 by peatland zone")
plt.savefig(save_path + "LAI_4ring_winter_peatZone.pdf")

###Boxplot of LAI5 values by zone
fig, [ax1, ax2] = plt.subplots(ncols=1, nrows=2, figsize=(3.5, 5.5),
                        layout="constrained")

#S2 LAI5
sns.boxplot(data = s2LAI_winter_grouped, x = 'Zone', y = 'OLS Prediction Ring 5',
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax1)
ax1.set_ylabel('LAI Ring 5')
at = offsetbox.AnchoredText(
    "S2", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)

#S6 LAI4
sns.boxplot(data = s6LAI_winter_grouped, x = 'Zone', y = 'OLS Prediction Ring 5',
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax2)
ax2.set_ylabel('LAI Ring 5')
at = offsetbox.AnchoredText(
    "S6", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)

plt.suptitle("LAI Ring 5 in S2,S6 by peatland zone")
plt.savefig(save_path + "LAI_5ring_winter_peatZone.pdf")

###Boxplot of LAI by peatland
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 3.5),
                        layout="constrained")

sns.boxplot(data = pd.concat([s2LAI_winter_grouped, s6LAI_winter_grouped], keys = ['S2', 'S6']).reset_index(),
    x = 'level_0', y = 'OLS Prediction Ring 5',
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "red"},
    ax = ax)
ax.set_ylabel('LAI Ring 5')
ax.set_xlabel('Watershed')
at = offsetbox.AnchoredText(
    "S2", prop=dict(size=15), frameon=True, loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)

plt.suptitle("LAI Ring 5 by Peatland")
plt.savefig(save_path + "LAI_ring5_peat_winter.pdf")

#%%

'''STATISTICAL ANALYSIS - SUMMER'''
#Sort into sets
s2upland = s2LAI_grouped[s2LAI_grouped.Zone == 'Upland']
s2lagg = s2LAI_grouped[s2LAI_grouped.Zone == 'Lagg']
s2bog = s2LAI_grouped[s2LAI_grouped.Zone == 'Bog']

s6upland = s6LAI_grouped[s6LAI_grouped.Zone == 'Upland']
s6lagg = s6LAI_grouped[s6LAI_grouped.Zone == 'Lagg']
s6bog = s6LAI_grouped[s6LAI_grouped.Zone == 'Bog']

###One-way ANOVA on Zones
#Run
s2_ANOVA = scipy.stats.f_oneway(s2upland['LAI 5Ring'], 
    s2bog['LAI 5Ring'], 
    s2lagg['LAI 5Ring'])

s6_ANOVA = scipy.stats.f_oneway(s6upland['LAI 5Ring'], 
    s6bog['LAI 5Ring'], 
    s6lagg['LAI 5Ring'])

#Print results
print(s2_ANOVA)
print(s6_ANOVA)

###Tukey test on Zones
#Run
s2_Tukey = scipy.stats.tukey_hsd(s2upland['LAI 5Ring'], 
    s2bog['LAI 5Ring'], 
    s2lagg['LAI 5Ring'])

s6_Tukey = scipy.stats.tukey_hsd(s6upland['LAI 5Ring'], 
    s6bog['LAI 5Ring'], 
    s6lagg['LAI 5Ring'])

#Print results
print(s2_Tukey)
print(s6_Tukey)

###One-way ANOVA on Zones
#Run
watershed_ANOVA = scipy.stats.f_oneway(s2LAI_grouped['LAI 5Ring'], 
    s6LAI_grouped['LAI 5Ring'])

#Print
print(watershed_ANOVA)

# %%

'''STAISTICAL ANALYSIS - WINTER'''
#Sort into sets
s2upland_winter = s2LAI_winter_grouped[s2LAI_winter_grouped.Zone == 'Upland']
s2lagg_winter = s2LAI_winter_grouped[s2LAI_winter_grouped.Zone == 'Lagg']
s2bog_winter = s2LAI_winter_grouped[s2LAI_winter_grouped.Zone == 'Bog']

s6upland_winter = s6LAI_winter_grouped[s6LAI_winter_grouped.Zone == 'Upland']
s6lagg_winter = s6LAI_winter_grouped[s6LAI_winter_grouped.Zone == 'Lagg']
s6bog_winter = s6LAI_winter_grouped[s6LAI_winter_grouped.Zone == 'Bog']

###One-way ANOVA on Zones
#Run
s2_ANOVA = scipy.stats.f_oneway(s2upland_winter['OLS Prediction Ring 5'], 
    s2bog_winter['OLS Prediction Ring 5'], 
    s2lagg_winter['OLS Prediction Ring 5'])

s6_ANOVA = scipy.stats.f_oneway(s6upland_winter['OLS Prediction Ring 5'], 
    s6bog_winter['OLS Prediction Ring 5'], 
    s6lagg_winter['OLS Prediction Ring 5'])

#Print results
print(s2_ANOVA)
print(s6_ANOVA)


###Tukey test on Zones
#Run
s2_Tukey = scipy.stats.tukey_hsd(s2upland_winter['OLS Prediction Ring 5'], 
    s2bog_winter['OLS Prediction Ring 5'], 
    s2lagg_winter['OLS Prediction Ring 5'])

s6_Tukey = scipy.stats.tukey_hsd(s6upland_winter['OLS Prediction Ring 5'], 
    s6bog_winter['OLS Prediction Ring 5'], 
    s6lagg_winter['OLS Prediction Ring 5'])

#Print results
print(s2_Tukey)
print(s6_Tukey)


###One-way ANOVA on Zones
#Run
watershed_ANOVA = scipy.stats.f_oneway(s2LAI_winter_grouped['OLS Prediction Ring 5'], 
    s6LAI_winter_grouped['OLS Prediction Ring 5'])

#Print
print(watershed_ANOVA)

###One-way ANOVA on Zones
bog_ANOVA = scipy.stats.f_oneway(s2bog_winter['OLS Prediction Ring 5'], 
    s6bog_winter['OLS Prediction Ring 5'])

print(bog_ANOVA)

lagg_ANOVA = scipy.stats.f_oneway(s2lagg_winter['OLS Prediction Ring 5'], 
    s6lagg_winter['OLS Prediction Ring 5'])

print(lagg_ANOVA)

upland_ANOVA = scipy.stats.f_oneway(s2upland_winter['OLS Prediction Ring 5'], 
    s6upland_winter['OLS Prediction Ring 5'])

print(upland_ANOVA)

# %%
'''STAISTICAL ANALYSIS - SUMMER vs. WINTER'''

###One-way ANOVA on Zones
#S2
s2upland_ANOVA = scipy.stats.f_oneway(s2upland['LAI 5Ring'], 
    s2upland_winter['OLS Prediction Ring 5'])
print(s2upland_ANOVA)

s2lagg_ANOVA = scipy.stats.f_oneway(s2lagg['LAI 5Ring'], 
    s2lagg_winter['OLS Prediction Ring 5'])
print(s2lagg_ANOVA)

s2bog_ANOVA = scipy.stats.f_oneway(s2bog['LAI 5Ring'], 
    s2bog_winter['OLS Prediction Ring 5'])
print(s2bog_ANOVA)

#S6
s6upland_ANOVA = scipy.stats.f_oneway(s6upland['LAI 5Ring'], 
    s6upland_winter['OLS Prediction Ring 5'])
print(s6upland_ANOVA)

s6lagg_ANOVA = scipy.stats.f_oneway(s6lagg['LAI 5Ring'], 
    s6lagg_winter['OLS Prediction Ring 5'])
print(s6lagg_ANOVA)

s6bog_ANOVA = scipy.stats.f_oneway(s6bog['LAI 5Ring'], 
    s6bog_winter['OLS Prediction Ring 5'])
print(s6bog_ANOVA)

# %%
#Plotting Specifics
custom_col = sns.color_palette(['#1b9e77', '#d95f02', '#7570b3'])
custom_pal = {'Upland': '#1b9e77', 
              'Lagg': '#d95f02', 
              'Bog': '#7570b3'}

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 3.5),
                        layout="constrained")

sns.boxplot(data = pd.concat([s2LAI_winter_grouped, s6LAI_winter_grouped], keys = ['S2', 'S6']).reset_index(),
    x = 'level_0', y = 'OLS Prediction Ring 5',
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    #boxprops={"facecolor": (.4, .6, .8, .5)},
    #medianprops={"color": "red"},
    hue = 'Zone',
    palette = custom_pal,
    ax = ax)

#ANOVA labels
ax.text(-0.27, 3.0, 'A', fontweight = 'bold', horizontalalignment = 'center')
ax.text(0, 3.0, 'B', fontweight = 'bold', horizontalalignment = 'center')
ax.text(0.27, 3.0, 'B', fontweight = 'bold', horizontalalignment = 'center')
ax.text(0.73, 3.0, 'B', fontweight = 'bold', horizontalalignment = 'center')
ax.text(1.0, 3.0, 'B', fontweight = 'bold', horizontalalignment = 'center')
ax.text(1.27, 3.0, 'B', fontweight = 'bold', horizontalalignment = 'center')


ax.set_ylabel('LAI Ring 5')
ax.set_xlabel('Watershed')
ax.set_ylim(0, 3.5)
ax.legend(bbox_to_anchor = (1.3, 1))



# %%
