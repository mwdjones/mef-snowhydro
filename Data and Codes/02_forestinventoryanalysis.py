# -*- coding: utf-8 -*-
"""
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
'''Import data'''
import_link = './Cleaned Data/'
forestInv = pd.read_excel(import_link + 'S2overstory_2023_compiled.xlsx')

# %%
'''Abundance Plots'''
forestInv_counts = forestInv.groupby(['NORTHING', 'EASTING', 'SPECIES'], as_index = False).count()

sns.relplot(forestInv_counts.EASTING, forestInv_counts.NORTHING,
            size = forestInv_counts.SITE, sizes = (100,500),
            col = forestInv_counts.SPECIES, col_wrap = 4, 
            height = 5, aspect = 1)
plt.show()

# %%
