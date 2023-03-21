# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:09:00 2023


@author: M.W.Jones
"""
#%%
'''PACKAGE IMPORTS'''
### for data analysis
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import seaborn as sns

### for custom legends
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from rasterio.plot import show
from matplotlib.colors import LinearSegmentedColormap

### geopandas, for dealing with shapefiles as Pandas dataframes
import geopandas as gpd
### contextily, for base maps
import contextily as cx
### rasterio for DEM
import osgeo
import rasterio


#%%
'''Import Marcell Shapefiles'''

#Import watershed shapefiles
mef = gpd.read_file("E:/1_DesktopBackup/Feng Research/Data/MEF GIS Data/watersheds/S2S6.shp")

#Import veg stakes dataframe - covert to geopandas
veg_stakes_data = pd.read_csv("E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/Data Collection/0 _ Experimental Setup/Vegetations Sampling Grid/GridCoordinates.csv")
veg_stakes = gpd.GeoDataFrame(
    veg_stakes_data, 
    geometry = gpd.points_from_xy(veg_stakes_data.NORTHING, veg_stakes_data.EASTING), 
    crs="EPSG:32633")

#Change MEF shapefiles projection
mef_proj = mef.to_crs('EPSG:32633')

#Import DEM
dem = rasterio.open("E:/1_DesktopBackup/Feng Research/Data/MEF Lidar/dem_mef.tif")

#%%
'''Subset to S2'''

#Clip spatial data to S2
veg_stakes_s2 = veg_stakes.cx[:, 5262200:5262600]
#%%
'''Plot'''

custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#1b9e77', '#d95f02', '#7570b3'], N = 3)

fig, ax = plt.subplots()

#extent=[veg_stakes_s2.total_bounds[0], veg_stakes_s2.total_bounds[2], veg_stakes_s2.total_bounds[1], veg_stakes_s2.total_bounds[3]]
ax = rasterio.plot.show(dem, alpha = 0.3,
                        ax=ax, cmap = 'Greens')
ax = rasterio.plot.show(dem, 
                        contour = True,
                        ax=ax, cmap = 'Greens')

veg_stakes_s2.plot(column = 'ZONE', k=3, categorical=True,
                   cmap = custom_cmap, ax=ax, 
                   zorder = 7, 
                   s = 80)

#Stake labels
#for idx,row in veg_stakes.iterrows():
#    x = row[2]
#    y = row[3]
#    lab = row[0]
#    plt.text(x, y+15, lab, va='baseline', ha='center')

ax.set_ylim(5262100.0, 5262700.0)
ax.set_ylabel('Easting')
ax.set_xlim(464300.0, 464900.0)
ax.set_xlabel('Northing')

plt.savefig('E:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Figures/laiPlots/mapPlot.pdf')

#%%