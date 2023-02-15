# -*- coding: utf-8 -*-
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
filepath = "D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/Data and Codes/Raw Data/"
save_path = 'D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/Figures/laiPlots/'
s2LAI_import = pd.read_csv(filepath + "S2_summerLAI.txt", sep = ';')
s6LAI_import = pd.read_csv(filepath + "S6_summerLAI.txt", sep = ';')

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

#%%
'''PLOTS ANALYSIS'''

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

'''STATISTICAL ANALYSIS'''
#Trim zone names
s2LAI_grouped.Zone =[col.strip() for col in s2LAI_grouped.Zone]
s6LAI_grouped.Zone =[col.strip() for col in s6LAI_grouped.Zone]

#Sort into sets
s2upland = s2LAI_grouped[s2LAI_grouped.Zone == 'Upland']
s2lagg = s2LAI_grouped[s2LAI_grouped.Zone == 'Lagg']
s2bog = s2LAI_grouped[s2LAI_grouped.Zone == 'Bog']

s6upland = s6LAI_grouped[s6LAI_grouped.Zone == 'Upland']
s6lagg = s6LAI_grouped[s6LAI_grouped.Zone == 'Lagg']
s6bog = s6LAI_grouped[s6LAI_grouped.Zone == 'Bog']

###One-way ANOVA on Zones
#Run
s2_ANOVA = scipy.stats.f_oneway(s2upland['LAI 4Ring'], 
    s2bog['LAI 4Ring'], 
    s2lagg['LAI 4Ring'])

s6_ANOVA = scipy.stats.f_oneway(s6upland['LAI 4Ring'], 
    s6bog['LAI 4Ring'], 
    s6lagg['LAI 4Ring'])

#Print results
print(s2_ANOVA)
###
#F_onewayResult(statistic=4.578213684927458, pvalue=0.019796661890366617) -- Significant at 0.05
###
print(s6_ANOVA)
###
#F_onewayResult(statistic=1.084076093212929, pvalue=0.35727696301072553) -- Not significant at 0.05
###

###Tukey test on Zones
#Run
s2_Tukey = scipy.stats.tukey_hsd(s2upland['LAI 4Ring'], 
    s2bog['LAI 4Ring'], 
    s2lagg['LAI 4Ring'])

s6_Tukey = scipy.stats.tukey_hsd(s6upland['LAI 4Ring'], 
    s6bog['LAI 4Ring'], 
    s6lagg['LAI 4Ring'])

#Print results
print(s2_Tukey)
###
# 0-1 and 1-0 (i.e. upland and bog) are signifcant at 0.05
#Tukey's HSD Pairwise Group Comparisons (95.0% Confidence Interval)
#Comparison  Statistic  p-value  Lower CI  Upper CI
# (0 - 1)      0.874     0.021     0.117     1.631
# (0 - 2)      0.511     0.233    -0.246     1.267
# (1 - 0)     -0.874     0.021    -1.631    -0.117
# (1 - 2)     -0.363     0.595    -1.283     0.557
# (2 - 0)     -0.511     0.233    -1.267     0.246
# (2 - 1)      0.363     0.595    -0.557     1.283
###
print(s6_Tukey)
###
# No significance (Expected no significance)
#Tukey's HSD Pairwise Group Comparisons (95.0% Confidence Interval)
#Comparison  Statistic  p-value  Lower CI  Upper CI
# (0 - 1)     -0.202     0.758    -0.918     0.514
# (0 - 2)     -0.350     0.343    -0.968     0.268
# (1 - 0)      0.202     0.758    -0.514     0.918
# (1 - 2)     -0.148     0.889    -0.956     0.660
# (2 - 0)      0.350     0.343    -0.268     0.968
# (2 - 1)      0.148     0.889    -0.660     0.956
##

###One-way ANOVA on Zones
#Run
watershed_ANOVA = scipy.stats.f_oneway(s2LAI_grouped['LAI 4Ring'], 
    s6LAI_grouped['LAI 4Ring'])

#Print
print(watershed_ANOVA)
###
# F_onewayResult(statistic=1.0373695573879274, pvalue=0.3133371801498456) - Not significant at 0.05
###

# %%
