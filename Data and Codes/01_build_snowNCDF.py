#%%
'''Imports'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd

#%%

'''Data'''
#Add data for each sampling date
data_s6 = [[[np.nan, 0, 0, np.nan, np.nan, np.nan],
            [0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0],
            [np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0]],
            [[np.nan, 5.5, 8, np.nan, np.nan, np.nan],
            [0, 3, 0, 5, 5, np.nan],
            [0, 7.5, 15, 0, 0, 0],
            [np.nan, 2, 6, 10, 0, 0],
            [np.nan, np.nan, 0, 9.5, 11, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0]], 
            [[np.nan, 31, 38, np.nan, np.nan, np.nan],
            [28, 33, 28, 24, 35, np.nan],
            [32, 39, 34, 21, 19, 18],
            [np.nan, 32, 31, 40, 19, 21],
            [np.nan, np.nan, 22, 26, 41, 27],
            [np.nan, np.nan, np.nan, 18, 29, 15]], 
            [[np.nan, 29, 32, np.nan, np.nan, np.nan],
            [20, 36, 26, 18, 30, np.nan],
            [25, 33.5, 39, 23, 20, 19],
            [np.nan, 28.5, 32, 30, 21, 22.5],
            [np.nan, np.nan, 20, 34.5, 36, 26],
            [np.nan, np.nan, np.nan, 21, 24, 19]], 
            [[np.nan, 32, 31, np.nan, np.nan, np.nan],
            [25, 28, 28, 25, np.nan, np.nan],
            [26, 32, 40, 22, 23, 19],
            [np.nan, 29, 29, 40, 20, 20],
            [np.nan, np.nan, 22, 31, 38, 29],
            [np.nan, np.nan, np.nan, 22, 26, 18]], 
            [[np.nan, 32.5, 36.5, np.nan, np.nan, np.nan],
            [27.5, 36, 25, 27, 28, np.nan],
            [28, 36, 40, 26, 24, 22],
            [np.nan, 34, 35, 31, 23, 23],
            [np.nan, np.nan, 22, 31.5, 38, 30],
            [np.nan, np.nan, np.nan, 25, 27, 20.5]], 
            [[np.nan, 28, 37, np.nan, np.nan, np.nan],
            [29, 37, 33, 32, 34, np.nan],
            [34, 43, 45, 30, 26, 22],
            [np.nan, 37, 39, 29, 26, 24],
            [np.nan, np.nan, 30, 38, 39, 29],
            [np.nan, np.nan, np.nan, 25, 32, 22]], 
            [[np.nan, 34, 35, np.nan, np.nan, np.nan],
            [27, 35, 31, 28, 36, np.nan],
            [28, 40, 47, 25, 26, 23],
            [np.nan, 32, 36, 43, 23, 27],
            [np.nan, np.nan, 24, 37, 41, 33],
            [np.nan, np.nan, np.nan, 22, 32, 23]], 
            [[np.nan, 27, 33, np.nan, np.nan, np.nan],
            [27, 27, 32, 27, 34, np.nan],
            [26, 35, 52, 23, 23, 21],
            [np.nan, 36, 30, 44, 18, 24],
            [np.nan, np.nan, 20, 25, 43, 26],
            [np.nan, np.nan, np.nan, 16, 11, 20]], 
            [[np.nan, 35, 41, np.nan, np.nan, np.nan],
            [34, 41, 32, 29, 42, np.nan],
            [25, 44, 54, 30, 28, 26],
            [np.nan, 33, 40, 48, 18, 30],
            [np.nan, np.nan, 26, 40, 59, 25],
            [np.nan, np.nan, np.nan, 23, 33, 25]], 
            [[np.nan, 41, 43, np.nan, np.nan, np.nan],
            [31, 29, 30, 30, 30, np.nan],
            [26, 42, 55, 25, 26, 26],
            [np.nan, 35, 26.5, 37, 21, 28],
            [np.nan, np.nan, 23, 48, 50, 31],
            [np.nan, np.nan, np.nan, 20, 35.5, 25.5]], 
            [[np.nan, 47, 53, np.nan, np.nan, np.nan],
            [42, 47, 45, 43, 53, np.nan],
            [32, 49, 65, 34, 37, 34],
            [np.nan, 31, 49, 45, 26, 41],
            [np.nan, np.nan, 33, 56, 61, 33],
            [np.nan, np.nan, np.nan, 29, 44, 34]], 
            [[np.nan, 60, 57, np.nan, np.nan, np.nan],
            [38, 39, 48, 51, 57, np.nan],
            [34, 37, 71, 38, 40, 40],
            [np.nan, 38, 52, 72, 29, 47],
            [np.nan, np.nan, 25, 56, 66, 22],
            [np.nan, np.nan, np.nan, 33, 53, 40]], 
            [[np.nan, 47, 51, np.nan, np.nan, np.nan],
            [34, 36, 32, 46, 52, np.nan],
            [29, 44, 57, 33, 25, 37],
            [np.nan, 47, 43, 56, 34, 40],
            [np.nan, np.nan, 23, 53, 62, 29],
            [np.nan, np.nan, np.nan, 30, 52, 35]], 
            [[np.nan, 21, 9, np.nan, np.nan, np.nan],
            [0, 14, 0, 9, 19, np.nan],
            [0, 10, 0, 0, 0, 1], #Need to interpolate S634 (likely 0)
            [np.nan, 0, 9, 35, 0, 0],
            [np.nan, np.nan, 0, 6, 35, 0],
            [np.nan, np.nan, np.nan, 0, 7, 0]], 
            [[np.nan, 8, 0, np.nan, np.nan, np.nan],
            [0, 0, 0, 0, 7, np.nan],
            [0, 0, 17, 0, 0, 0],
            [np.nan, 0, 0, 28, 0, 0],
            [np.nan, np.nan, 0, 3, 16, 0],
            [np.nan, np.nan, np.nan, 0, 3, 0]], 
            [[np.nan, 0, 0, np.nan, np.nan, np.nan],
            [0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0],
            [np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0]], 
            [[np.nan, 0, 0, np.nan, np.nan, np.nan],
            [0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0],
            [np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0]], 
            [[np.nan, 9, 11, np.nan, np.nan, np.nan],
            [9, 12, 9, 10, 9, np.nan],
            [9, 10, 14, 9, 7, 8],
            [np.nan, 10, 10, 15, 7, 7],
            [np.nan, np.nan, 7, 9, np.nan, 7],
            [np.nan, np.nan, np.nan, 6, 7, 3]], 
            [[np.nan, 6, 7, np.nan, np.nan, np.nan],
            [4, 5, 6, 6, 6, np.nan],
            [3, 8, 9, 4, 4, 4],
            [np.nan, 6, 8, 7, 1, 4],
            [np.nan, np.nan, 3, 7, 9, 2],
            [np.nan, np.nan, np.nan, 2, 4, 1]], 
            [[np.nan, 0.5, 0.5, np.nan, np.nan, np.nan],
            [0.5, 0.5, 0.5, 0.5, 0.5, np.nan],
            [0.5, 0.5, 2, 0.5, 0.5, 0.5],
            [np.nan, 0.5, 0.5, 1, 0.5, 0.5],
            [np.nan, np.nan, 0.5, 0.5, 0.5, 0.5],
            [np.nan, np.nan, np.nan, 0.5, 0.5, 0.5]], 
            [[np.nan, 0.5, 0.5, np.nan, np.nan, np.nan],
            [0.5, 0.5, 0.5, 0.5, 0.5, np.nan],
            [0.5, 0.5, 2, 0.5, 0.5, 0.5],
            [np.nan, 0.5, 0.5, 1, 0.5, 0.5],
            [np.nan, np.nan, 0.5, 0.5, 0.5, 0.5],
            [np.nan, np.nan, np.nan, 0.5, 0.5, 0.5]], 
            [[np.nan, 1, 2, np.nan, np.nan, np.nan],
            [0.5, 0.5, 0.5, 1, 1, np.nan],
            [0.5, 1, 4, 0.5, 1, 0.5],
            [np.nan, 1, 0.5, 4, 0.5, 0.5],
            [np.nan, np.nan, 0.5, 0.5, 2, 0.5],
            [np.nan, np.nan, np.nan, 0.5, 0.5, 0.5]], 
            [[np.nan, 0.5, 0, np.nan, np.nan, np.nan],
            [0, 0, 0, 0, 0, np.nan],
            [0, 0, 2, 0, 0, 0.5],
            [np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0.5, 2, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0]], 
            [[np.nan, 0.5, 0, np.nan, np.nan, np.nan],
            [0.5, 0, 0, 0, 0, np.nan],
            [0, 0.5, 0.5, 0, 0, 0.5],
            [np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0.5, 0.5, 0.5, 0],
            [np.nan, np.nan, np.nan, 0, 0.5, 0.5]], 
            [[np.nan, 0, 0, np.nan, np.nan, np.nan],
            [0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0],
            [np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0]], 
            [[np.nan, 0, 0, np.nan, np.nan, np.nan],
            [0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0],
            [np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0]], 
            [[np.nan, 0, 0, np.nan, np.nan, np.nan],
            [0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0],
            [np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0]], 
            [[np.nan, 17, 17, np.nan, np.nan, np.nan],
            [20, 17, 18, 19, 16, np.nan],
            [18, 18, 21, 16, 17, 14],
            [np.nan, 18, 17, 22, 15, 17],
            [np.nan, np.nan, 16, 18, 18, 14],
            [np.nan, np.nan, np.nan, 17, 19, 15]], 
            [[np.nan, 8, 6, np.nan, np.nan, np.nan],
            [8, 6, 9, 11, 10, np.nan],
            [6, 10, 9, 5, 9, 10],
            [np.nan, 8, 10, 12, 5, 10],
            [np.nan, np.nan, 8, 12, 12, 5],
            [np.nan, np.nan, np.nan, 9, 14, 8]], 
            [[np.nan, 0, 0, np.nan, np.nan, np.nan],
            [0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0],
            [np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0]], 
            [[np.nan, 0, 0, np.nan, np.nan, np.nan],
            [0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0],
            [np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0]], 
            [[np.nan, 9, 10, np.nan, np.nan, np.nan], #12/04/2024
            [11, 11, 8, 9, 10, np.nan],
            [7, 10, 15, 9, 9, 7],
            [np.nan, 10, 10, 9, 7, 8],
            [np.nan, np.nan, 7, 8, 10, 8],
            [np.nan, np.nan, np.nan, 6, 8, 5]], 
            [[np.nan, 12, 16, np.nan, np.nan, np.nan], #12/20/2024
            [10, 16, 15, 14, 16, np.nan],
            [14, 17, 23, 10, 8, 9],
            [np.nan, 17, 15, 22, 8, 7],
            [np.nan, np.nan, 7, 13, 11, 11],
            [np.nan, np.nan, np.nan, 9, 12, 8]], 
            [[np.nan, 10, 12, np.nan, np.nan, np.nan], #01/06/2025
            [6, 11, 10, 10, 10, np.nan],
            [10, 11, 18, 8, 8, 7],
            [np.nan, 15, 11, 15, 5, 8],
            [np.nan, np.nan, 7, 10, 12, 7],
            [np.nan, np.nan, np.nan, 5, 8, 5]], 
            [[np.nan, 10, 12, np.nan, np.nan, np.nan], #01/10/2025
            [8, 14, 11, 11, 10, np.nan],
            [10, 13, 18, 10, 11, 8],
            [np.nan, 18, 12, 19, 6, 10],
            [np.nan, np.nan, 7, 12, 21, 7],
            [np.nan, np.nan, np.nan, 6, 8, 6]],
            [[np.nan, 15, 20, np.nan, np.nan, np.nan], #01/17/2025
            [13, 17, 14, 17, 16, np.nan],
            [15, 16, 25, 12, 12, 11],
            [np.nan, 24, 18, 22, 10, 13],
            [np.nan, np.nan, 11, 16, 23, 12],
            [np.nan, np.nan, np.nan, 10, 13, 9]], 
            [[np.nan, 20, 23, np.nan, np.nan, np.nan], #01/24/2025
            [18, 20, 18, 19, 22, np.nan],
            [20, 23, 32, 17, 16, 15],
            [np.nan, 23, 22, 22, 13, 15],
            [np.nan, np.nan, 13, 17, 27, 13],
            [np.nan, np.nan, np.nan, 12, 16, 13]], 
            [[np.nan, 19, 23, np.nan, np.nan, np.nan], #01/30/2025
            [11, 19, 16, 16, 19, np.nan],
            [19, 20, 29, 15, 15, 12],
            [np.nan, 25, 23, 23, 9, 15],
            [np.nan, np.nan, 11, 18, 26, 14],
            [np.nan, np.nan, np.nan, 10, 17, 11]], 
            [[np.nan, 28, 37, np.nan, np.nan, np.nan], #02/07/2025
            [24, 32, 30, 30, 32, np.nan],
            [29, 31, 35, 25, 24, 23],
            [np.nan, 38, 33, 31, 23, 22],
            [np.nan, np.nan, 19, 29, 41, 24],
            [np.nan, np.nan, np.nan, 20, 25, 20]], 
            [[np.nan, 29, 33, np.nan, np.nan, np.nan], #02/14/2025
            [21, 29, 27, 26, 29, np.nan],
            [26, 30, 36, 23, 23, 23],
            [np.nan, 34, 33, 44, 18, 23],
            [np.nan, np.nan, 18, 30, 35, 22],
            [np.nan, np.nan, np.nan, 15, 23, 19]],
            [[np.nan, 27, 34, np.nan, np.nan, np.nan], #2/21/2025
            [24, 30, 28, 28, 33, np.nan],
            [26, 32, 36, 26, 24, 21],
            [np.nan, 40, 34, 38, 20, 25],
            [np.nan, np.nan, 19, 30, 35, 21],
            [np.nan, np.nan, np.nan, 19, 25, 22]], 
            [[np.nan, 12, 16, np.nan, np.nan, np.nan], #03/07/2025
            [7, 11, 12, 7, 12, np.nan],
            [7, 14, 22, 6, 6, 10],
            [np.nan, 20, 21, 20, 3, 13],
            [np.nan, np.nan, 3, 14, 26, 1],
            [np.nan, np.nan, np.nan, 9.5, 11, 7]], 
            [[np.nan, 7, 10, np.nan, np.nan, np.nan], #3/13/2025
            [0, 0, 0, 3, 13, np.nan],
            [0, 8, 23, 0, 0, 3],
            [np.nan, 13, 9, 17, 0, 4],
            [np.nan, np.nan, 0, 11, 18, 0],
            [np.nan, np.nan, np.nan, 0, 2, 0]],
            [[np.nan, 4, 2, np.nan, np.nan, np.nan], #03/20/2025
            [0, 0, 0, 0, 5, np.nan],
            [0, 0.5, 17, 0, 0, 1],
            [np.nan, 4, 1, 13, 0, 0.5],
            [np.nan, np.nan, 0.5, 1, 16, 0],
            [np.nan, np.nan, np.nan, 0, 0.5, 1]],
            [[np.nan, 3, 0, np.nan, np.nan, np.nan], #03/27/2025
            [0, 0, 0, 0, 3, np.nan],
            [0, 0, 10, 0, 0, 0.5],
            [np.nan, 3, 0, 20, 0, 0],
            [np.nan, np.nan, 0, 0.5, 10, 0],
            [np.nan, np.nan, np.nan, 0, 0.5, 0]],
            [[np.nan, 7, 9, np.nan, np.nan, np.nan], #04/04/2025
            [9, 10, 8, 7, 10, np.nan],
            [8, 7, 12, 7, 8, 6],
            [np.nan, 8, 7, 19, 5, 8],
            [np.nan, np.nan, 6, 8, 11, 8],
            [np.nan, np.nan, np.nan, 8, 10, 5]],
            [[np.nan, 5, 0, np.nan, np.nan, np.nan], #04/10/2025
            [0, 0, 0, 3, 5, np.nan],
            [0, 3, 7, 0, 1, 2],
            [np.nan, 3, 2, 18, 0, 4],
            [np.nan, np.nan, 0.5, 2, 8, 0],
            [np.nan, np.nan, np.nan, 0.5, 5, 1]]]

data_s2 = [[[0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]],
            [[5, 10, 0, 8.5, 0, 3, np.nan],
            [np.nan, 11, 5, 19, 13, 11, np.nan], # S214 interpolated (13)
            [14.5, 5, 14, 14.5, 5, 8, 8.5],
            [np.nan, np.nan, 9, 10, 9.5, 14, 11],
            [np.nan, np.nan, np.nan, 5.5, 8, 9, 4],
            [np.nan, np.nan, np.nan, np.nan, 12, 5, np.nan]], 
            [[19.25, 24, 24, 32, 23.5, 25, np.nan], #19.25 interpolated in time
            [np.nan, 26, 22.5, 28, 32, 44, np.nan], #28 interpolated in time
            [47, 16, 29, 34, 31, 34, 35], #16 interpolated in time
            [np.nan, np.nan, 26, 32, 46, 37, 29],
            [np.nan, np.nan, np.nan, 36.5, 27.5, 33, 35],
            [np.nan, np.nan, np.nan, np.nan, 24.5, 24, np.nan]], 
            [[33.5, 31, 25, 45, 21, 34, np.nan],
            [np.nan, 34, 27.5, 37, 34.5, 41.5, np.nan],
            [33.5, 27, 41, 42, 36, 40.5, 33.5],
            [np.nan, np.nan, 36.5, 36, 38, 46.5, 37],
            [np.nan, np.nan, np.nan, 35, 35.5, 37, 35],
            [np.nan, np.nan, np.nan, np.nan, 33, 32.5, np.nan]], 
            [[37, 32, 22, 38, 7, 26, np.nan],
            [np.nan, 38, 29, 41, 31, 35, np.nan],
            [29, 26, 38.5, 33, 33, 45, 34],
            [np.nan, np.nan, 32, 28, 35, 33, 33],
            [np.nan, np.nan, np.nan, 34, 38, 36, 37],
            [np.nan, np.nan, np.nan, np.nan, 32, 30, np.nan]], 
            [[40, 40, 28.5, 29, 25, 33.5, np.nan],
            [np.nan, 43.5, 32, 47, 36, 43.5, np.nan],
            [40.5, 28.5, 41.5, 44.5, 46, 42, 40],
            [np.nan, np.nan, 37, 43, 36, 49.5, 42],
            [np.nan, np.nan, np.nan, 33.5, 40.5, 37.5, 36],
            [np.nan, np.nan, np.nan, np.nan, 38.5, 36, np.nan]], 
            [[41.5, 39.5, 27.75, 30.5, 25, 35.75, np.nan], #Whole timestamp is linearly interpolated from the previous and subsequent stakes
            [np.nan, 42.25, 34, 39, 37, 35.75, np.nan],
            [40.75, 29.75, 42.25, 45.25, 42.5, 49, 40],
            [np.nan, np.nan, 39, 44, 38.5, 45.75, 42.5],
            [np.nan, np.nan, np.nan, 38.25, 43.25, 40.25, 40],
            [np.nan, np.nan, np.nan, np.nan, 38.75, 37, np.nan]], 
            [[43, 39, 27, 32, 25, 38, np.nan],
            [np.nan, 41, 36, 31, 38, 48, np.nan],
            [41, 31, 43, 46, 39, 56, 40],
            [np.nan, np.nan, 41, 45, 41, 42, 43],
            [np.nan, np.nan, np.nan, 43, 46, 44, 44],
            [np.nan, np.nan, np.nan, np.nan, 39, 38, np.nan]], 
            [[29, 34, 26, 28, 15, 35, np.nan],
            [np.nan, 51, 28, 37, 36, 38, np.nan],
            [44, 34.5, 30, 37, 39, 47, 35], #34.5 interpolated in time
            [np.nan, np.nan, 39, 41, 49, 42, 40],
            [np.nan, np.nan, np.nan, 40, 45, 42, 40],
            [np.nan, np.nan, np.nan, np.nan, 37, 32, np.nan]], 
            [[45, 41, 32, 42, 31, 44, np.nan],
            [np.nan, 50, 41, 52, 30, 52, np.nan],
            [52, 38, 49, 52, 46, 53, 44],
            [np.nan, np.nan, 52, 49, 61, 56, 46],
            [np.nan, np.nan, np.nan, 44, 49, 48, 50],
            [np.nan, np.nan, np.nan, np.nan, 44, 46, np.nan]], 
            [[43, 48, 29.5, 29, 29, 39, np.nan],
            [np.nan, 39, 30, 62.5, 45.5, 22, np.nan],
            [53, 27, 46, 46, 45, 57, 41],
            [np.nan, np.nan, 51.5, 47, 54, 60, 48],
            [np.nan, np.nan, np.nan, 41.5, 47.5, 48.5, 48],
            [np.nan, np.nan, np.nan, np.nan, 46, 38, np.nan]], 
            [[53, 49, 33, 38, 36, 55, np.nan],
            [np.nan, 60, 51, 41, 58, 76, np.nan],
            [64, 49, 60, 59, 62, 61, 54],
            [np.nan, np.nan, 55, 60, 66, 68, 57],
            [np.nan, np.nan, np.nan, 60, 60, 52, 59],
            [np.nan, np.nan, np.nan, np.nan, 60, 53, np.nan]], 
            [[70, 59, 33, 38, 34, 62, np.nan],
            [np.nan, 57, 58, 78, 60, 73, np.nan],
            [68, 56, 63, 69, 60, 50, 59],
            [np.nan, np.nan, 66, 68, 70, 75, 56.5], #S236 interpolated (56.5)
            [np.nan, np.nan, np.nan, 66, 65, 55, 65],
            [np.nan, np.nan, np.nan, np.nan, 59, 57, np.nan]], 
            [[53, 45, 29, 49, 20, 57, np.nan],
            [np.nan, 64, 50, 76, 53, 59, np.nan],
            [44, 50, 57, 64, 64, 51, 53],
            [np.nan, np.nan, 53, 44, 62, 66, 54],
            [np.nan, np.nan, np.nan, 62, 60, 59, 57],
            [np.nan, np.nan, np.nan, np.nan, 48, 54, np.nan]], 
            [[0, 0, 0, 0, 0, 6, np.nan], #Need to interpolate S200 (likely 0)
            [np.nan, 7, 10, 21, 2, 10, np.nan],
            [0, 17, 26, 25, 35, 34, 0], #Need to interpolate S220 (likely 0)
            [np.nan, np.nan, 22, 18, 29, 26, 13],
            [np.nan, np.nan, np.nan, 15, 18, 9, 10],
            [np.nan, np.nan, np.nan, np.nan, 6, 5, np.nan]], 
            [[0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 6, 0, 0, np.nan],
            [0, 0, 11, 0, 21, 19, 0],
            [np.nan, np.nan, 19, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]], 
            [[0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]], 
            [[0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]], 
            [[11, 10, 13, 8, 6, 10, np.nan],
            [np.nan, 11, 5, 17, 11, 11, np.nan],
            [11, 8, 9, 14, 12, 12, 11],
            [np.nan, np.nan, 11, 11, 11, 13, 15],
            [np.nan, np.nan, np.nan, 11, 10, 12, 11],
            [np.nan, np.nan, np.nan, np.nan, 12, 11, np.nan]], 
            [[8, 7, 8, 5, 3, 9, np.nan],
            [np.nan, 9, 5, 7, 8, 6, np.nan],
            [7, 5, 10, 6, 7, 9, 8],
            [np.nan, np.nan, 8, 6, 8, 10, 9],
            [np.nan, np.nan, np.nan, 8, 9, 8, 6],
            [np.nan, np.nan, np.nan, np.nan, 9, 8, np.nan]], 
            [[1, 0.5, 0.5, 0.5, 0.5, 0.5, np.nan], # <1 cm amounts were entered as 0.5
            [np.nan, 2, 0.5, 2, 0.5, 0.5, np.nan],
            [0.5, 0.5, 2, 3, 0.5, 4, 0.5],
            [np.nan, np.nan, 1, 1, 6.5, 3, 2],
            [np.nan, np.nan, np.nan, 1, 1, 0.5, 0.5],
            [np.nan, np.nan, np.nan, np.nan, 1, 0.5, np.nan]], 
            [[1, 0.5, 0.5, 0.5, 0.5, 0.5, np.nan], # <1 cm amounts were entered as 0.5
            [np.nan, 2, 0.5, 2, 0.5, 0.5, np.nan],
            [0.5, 0.5, 2, 3, 0.5, 4, 0.5],
            [np.nan, np.nan, 1, 1, 6.5, 3, 2],
            [np.nan, np.nan, np.nan, 1, 1, 0.5, 0.5],
            [np.nan, np.nan, np.nan, np.nan, 1, 0.5, np.nan]], 
            [[2, 1, 2, 1, 0.5, 0.5, np.nan],
            [np.nan, 2, 0.5, 2, 1, 0.5, np.nan],
            [1, 1, 3, 1, 2, 2, 0.5],
            [np.nan, np.nan, 2, 1, 4, 4, 2],
            [np.nan, np.nan, np.nan, 3, 2, 2, 2],
            [np.nan, np.nan, np.nan, np.nan, 0.5, 2, np.nan]], 
            [[0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 2, 0, 1, 0, 0, np.nan],
            [0, 0, 2, 3, 1, 1, 0],
            [np.nan, np.nan, 2, 0.5, 4, 5, 1],
            [np.nan, np.nan, np.nan, 2, 3, 0.5, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]], 
            [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, np.nan],
            [np.nan, 0.5, 0.5, 0.5, 0.5, 0.5, np.nan],
            [0.5, 0.5, 1, 0.5, 0.5, 1, 0.5],
            [np.nan, np.nan, 0.5, 0.5, 1, 2, 1],
            [np.nan, np.nan, np.nan, 0.5, 0.5, 1, 0.5],
            [np.nan, np.nan, np.nan, np.nan, 0.5, 0.5, np.nan]], 
            [[0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]], 
            [[0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]], 
            [[0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]], 
            [[22, 18, 18, 12, 17, 20, np.nan],
            [np.nan, 18, 16, 20, 18, 21, np.nan],
            [18, 18, 20, 25, 20, 17, 16],
            [np.nan, np.nan, 20, 21, 15, 21, 18],
            [np.nan, np.nan, np.nan, 19, 15, 23, 20],
            [np.nan, np.nan, np.nan, np.nan, 18, 18, np.nan]], 
            [[10, 8, 7, 8, 8, 10, np.nan],
            [np.nan, 10, 9, 11, 7, 11, np.nan],
            [7, 9, 12, 10, 13, 8, 5],
            [np.nan, np.nan, 11, 8, 13, 10, 7],
            [np.nan, np.nan, np.nan, 9, 6, 12, 8],
            [np.nan, np.nan, np.nan, np.nan, 8, 6, np.nan]],
            [[0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]], 
            [[0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, np.nan],
            [0, 0, 0, 0, 0, 0, 0],
            [np.nan, np.nan, 0, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, 0, 0, 0, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]], 
            [[12, 10, 11, 9, 6, 10, np.nan], #12/4/2024
            [np.nan, 11, 7, 12, 10, 11, np.nan],
            [11, 8, 11, 8, 10, 10, 11],
            [np.nan, np.nan, 9, 9, 15, 11, 11],
            [np.nan, np.nan, np.nan, 10, 8, 11, 10],
            [np.nan, np.nan, np.nan, np.nan, 11, 9, np.nan]],
            [[15, 15, 20, 12, 7, 16, np.nan], #12/20/24
            [np.nan, 19, 11, 23, 16, 24, np.nan],
            [16, 13, 20, 22, 16, 22, 18],
            [np.nan, np.nan, 15, 18, 28, 23, 17],
            [np.nan, np.nan, np.nan, 19, 19, 20, 15],
            [np.nan, np.nan, np.nan, np.nan, 19, 9.5, np.nan]], #9.5 interpolated
            [[12, 11, 9, 10, 4, 12, np.nan], #1/6/2025
            [np.nan, 13, 9, 15, 10, 15, np.nan],
            [15, 10, 17, 16, 13, 15, 13],
            [np.nan, np.nan, 12, 13, 15, 22, 11],
            [np.nan, np.nan, np.nan, 12, 13, 15, 12],
            [np.nan, np.nan, np.nan, np.nan, 13, 10, np.nan]],
            [[14, 15, 11, 13, 6, 10, np.nan], #1/10/2025
            [np.nan, 13, 9, 21, 13, 16, np.nan],
            [14, 7, 18, 21, 14, 18, 16],
            [np.nan, np.nan, 16, 14, 22, 22, 14],
            [np.nan, np.nan, np.nan, 16, 14, 15, 16],
            [np.nan, np.nan, np.nan, np.nan, 17, 12, np.nan]],
            [[18, 16, 18, 13, 9, 21, np.nan], #1/17/2025
            [np.nan, 20, 16, 20, 17, 24, np.nan],
            [17, 10, 27, 26, 20, 19, 18],
            [np.nan, np.nan, 17, 22, 25, 27, 19],
            [np.nan, np.nan, np.nan, 18, 28, 21, 20],
            [np.nan, np.nan, np.nan, np.nan, 21, 17, np.nan]],
            [[21, 19, 24, 20, 11, 21, np.nan], #1/24/2025
            [np.nan, 25, 15, 25, 20, 28, np.nan],
            [24, 18, 26, 24, 23, 23, 22],
            [np.nan, np.nan, 22, 24, 28, 32, 24],
            [np.nan, np.nan, np.nan, 23, 23, 28, 23],
            [np.nan, np.nan, np.nan, np.nan, 25, 20, np.nan]],
            [[20, 18, 21, 17, 10, 20, np.nan], #1/30/25
            [np.nan, 24, 17, 27, 22, 25, np.nan],
            [19, 14, 27, 28, 17, 23, 22],
            [np.nan, np.nan, 21, 25, 24, 32, 24],
            [np.nan, np.nan, np.nan, 21, 21, 24, 21],
            [np.nan, np.nan, np.nan, np.nan, 23, 22, np.nan]],
            [[33, 31, 37, 33, 23, 35, np.nan], #2/7/2025
            [np.nan, 36, 27, 38, 33, 40, np.nan],
            [35, 28, 33, 39, 32, 38, 34],
            [np.nan, np.nan, 36, 36, 44, 49, 34],
            [np.nan, np.nan, np.nan, 37, 36, 33, 35],
            [np.nan, np.nan, np.nan, np.nan, 35, 35, np.nan]],
            [[32, 29, 33, 36, 18, 29, np.nan], #2/14/2025
            [np.nan, 35, 25, 31, 32, 37, np.nan],
            [33, 21, 37, 43, 32, 31, 30],
            [np.nan, np.nan, 34, 31, 41, 43, 33],
            [np.nan, np.nan, np.nan, 35, 34, 34, 30],
            [np.nan, np.nan, np.nan, np.nan, 34, 26, np.nan]], 
            [[33, 32, 36, 29, 18, 31, np.nan], #2/21/2025
            [np.nan, 35, 28, 37, 32, 37, np.nan],
            [34, 27, 36, 42, 35, 36, 36],
            [np.nan, np.nan, 34, 36, 40, 43, 35],
            [np.nan, np.nan, np.nan, 34, 38, 33, 33],
            [np.nan, np.nan, np.nan, np.nan, 36, 29, np.nan]],
            [[14, 9, 19, 14, 1, 14, np.nan], #3/7/2025
            [np.nan, 21, 13, 28, 16, 22, np.nan],
            [19, 15, 17, 26, 13, 19, 18],
            [np.nan, np.nan, 21, 16, 30, 28, 17],
            [np.nan, np.nan, np.nan, 20, 25, 26, 17],
            [np.nan, np.nan, np.nan, np.nan, 17, 18, np.nan]],
            [[10, 7, 15, 6, 0, 10, np.nan], #3/13/2025
            [np.nan, 18, 5, 25, 8, 10, np.nan],
            [9, 2, 21, 17, 13, 19, 18],
            [np.nan, np.nan, 11, 9, 21, 28, 17], #21 interp
            [np.nan, np.nan, np.nan, 15, 19, 11, 17],
            [np.nan, np.nan, np.nan, np.nan, 9, 8, np.nan]],
            [[0, 0, 0, 0, 0, 0.5, np.nan], #3/20/2025
            [np.nan, 5, 0, 11, 0.5, 0.5, np.nan],
            [4, 0, 6, 7, 8, 13, 0],
            [np.nan, np.nan, 7, 15, 12, 15, 1],
            [np.nan, np.nan, np.nan, 6, 6, 4, 0],
            [np.nan, np.nan, np.nan, np.nan, 0.5, 0.5, np.nan]],
            [[0, 0, 0, 0, 0, 0, np.nan], #3/27/2025
            [np.nan, 5, 0, 13, 0, 0, np.nan],
            [0, 0, 5, 10, 9, 0, 0],
            [np.nan, np.nan, 8, 9, 5, 14, 0],
            [np.nan, np.nan, np.nan, 0.5, 7, 7, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 4.75, np.nan]], #4.75 is interpolated
            [[0, 0, 0, 7, 7, 10, np.nan], #04/4/2025, 0s are interpolated
            [np.nan, 8, 8, 12, 9, 10, np.nan],
            [0, 8, 10, 10, 8, 9, 8], #0 is interpolated
            [np.nan, np.nan, 7, 7, 8, 15, 9],
            [np.nan, np.nan, np.nan, 8, 12, 10, 8],
            [np.nan, np.nan, np.nan, np.nan, 7, 10, np.nan]], 
            [[0, 0, 0, 0, 2, 2, np.nan], #04/10/2025
            [np.nan, 0, 2, 5, 0, 0, np.nan],
            [0, 4, 6, 10, 5, 5, 0],
            [np.nan, np.nan, 3, 7, 0, 8, 0],
            [np.nan, np.nan, np.nan, 0, 4, 0, 0],
            [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan]]]

'''Times'''
#Add dates as needed
times = np.array(['11-25-2022', '12-02-2022', '12-30-2022',
 '01-05-2023', '01-13-2023', '01-20-2023',
 '02-01-2023', '02-10-2023', '02-17-2023', '02-24-2023', 
 '03-09-2023', '03-17-2023', '03-24-2023', '03-31-2023', 
 '04-14-2023', '04-28-2023', '05-05-2023', '12-01-2023',
 '01-22-2024', '01-31-2024', '02-01-2024', '02-09-2024',
 '02-15-2024', '02-21-2024', '03-01-2024', '03-06-2024', 
 '03-13-2024', '03-21-2024', '03-29-2024', '04-03-2024', 
 '04-11-2024', '04-19-2024', 
 '12-04-2024', '12-20-2024', '01-06-2025', '01-10-2025', 
 '01-17-2025', '01-24-2025', '01-30-2025', '02-07-2025',
 '02-14-2024', '02-21-2025', '03-07-2025', '03-13-2025', 
 '03-20-2025', '03-27-2025', '04-04-2025', '04-10-2025']) 

'''Dims'''
#These are averaged from the stake coordinates to make the data fit a grid system
Northing_s6 = np.array(np.linspace(464644, 464335, 6))
Easting_s6 = np.array(np.linspace(5262240, 5263285, 6))

stakes_s6 = [[np.nan, -93.472876, -93.472239, np.nan, np.nan, np.nan],
            [-93.473668, -93.472898, -93.472114, -93.471344, -93.470533, np.nan],
            [-93.473742, -93.472985, -93.472175, -93.471365, -93.470515, -93.469744],
            [np.nan, -93.472927, -93.472211, -93.47128, -93.470324, -93.469633],
            [np.nan, np.nan, -93.472112, -93.471354, -93.472245, -93.473135],
            [np.nan, np.nan, np.nan, -93.471257, -93.470394, -93.469731]]

lat_s6 = [[np.nan, 47.521954, 47.52202, np.nan, np.nan, np.nan],
            [47.521429, 47.52145, 47.521481, 47.521466, 47.52146, np.nan],
            [47.520808, 47.520856, 47.520886, 47.520917, 47.520848, 47.520779],
            [np.nan, 47.520298, 47.520373, 47.520287, 47.520363, 47.520276],
            [np.nan, np.nan, 47.519735, 47.519693, 47.519752, 47.519821],
            [np.nan, np.nan, np.nan, 47.519244, 47.51922, 47.519331]]

lon_s6 = [[np.nan, 'S655', 'S654', np.nan, np.nan, np.nan],
            ['S646', 'S645', 'S644', 'S643', 'S642', np.nan],
            ['S636', 'S635', 'S634', 'S633', 'S632', 'S631'],
            [np.nan, 'S625', 'S624', 'S623', 'S622', 'S621'],
            [np.nan, np.nan, 'S614', 'S613', 'S612', 'S611'],
            [np.nan, np.nan, np.nan, 'S603', 'S602', 'S601']]

zones_s6 = [[np.nan, 'Upland', 'Upland', np.nan, np.nan, np.nan],
            ['Upland', 'Bog', 'Upland', 'Upland', 'Upland', np.nan], #s644 should be lagg - diff from Kristina LAI analysis
            ['Upland', 'Lagg', 'Bog', 'Lagg', 'Upland', 'Upland'],
            [np.nan, 'Upland', 'Lagg', 'Bog', 'Lagg', 'Upland'], #s621 should be lagg - diff from Kristina LAI analysis
            [np.nan, np.nan, 'Upland', 'Lagg', 'Bog', 'Lagg'],
            [np.nan, np.nan, np.nan, 'Upland', 'Upland', 'Upland']]

aspect_s6 = [[np.nan, 259.7763, 350.3666, np.nan, np.nan, np.nan],
            [166.1192, 154.9844, 226.2777, 203.7467, 188.4725, np.nan],
            [284.0857, 23.90763, 81.84194, 214.6792, 250.6175, 264.7998],
            [np.nan, 149.6178, 51.39898, 348.6811, 215.5392, 210.6501],
            [np.nan, np.nan, 84.29677, 37.4997, 161.5534, 84.30152],
            [np.nan, np.nan, np.nan, 100.015, 13.71132, 356.9422]]

slope_s6 = [[np.nan, 6.629145, 1.924451, np.nan, np.nan, np.nan],
            [3.132299, 0.592828, 13.37794, 2.9316, 1.701028, np.nan],
            [9.7621, 21.69698, 0.253367, 16.22974, 4.093031, 1.186282],
            [np.nan, 1.203789, 25.4478, 0.547875, 4.916601, 6.712477],
            [np.nan, np.nan, 1.439196, 19.39961, 0.339843, 0.71976],
            [np.nan, np.nan, np.nan, 0.618323, 10.60362, 4.687178]]


Northing_s2 = np.array(np.linspace(464819, 464448, 6)) 
Easting_s2 = np.array(np.linspace(5262240, 5262552, 7))

stakes_s2 = [['S200', 'S201', 'S202', 'S203', 'S204', 'S205', np.nan],
            [np.nan, 'S211', 'S212', 'S213', 'S214', 'S215', np.nan],
            ['S220', 'S221', 'S222', 'S223', 'S224', 'S225', 'S226'],
            [np.nan, np.nan, 'S232', 'S233', 'S234', 'S235', 'S236'],
            [np.nan, np.nan, np.nan, 'S243', 'S244', 'S245', 'S246'],
            [np.nan, np.nan, np.nan, np.nan, 'S254', 'S255', np.nan]]

lat_s2 = [[47.515147, 47.515294, 47.515423, 47.51539, 47.515177, 47.515216, np.nan],
            [np.nan, 47.514807, 47.514882, 47.51485, 47.514817, 47.514794, np.nan],
            [47.514264, 47.514223, 47.514388, 47.514256, 47.514313, 47.51429, 47.514302],
            [np.nan, np.nan, 47.513731, 47.51377, 47.513729, 47.513777, 47.513699],
            [np.nan, np.nan, np.nan, 47.513203, 47.513242, 47.513165, 47.513105],
            [np.nan, np.nan, np.nan, np.nan, 47.51264, 47.512634, np.nan]]

lon_s2 = [[-93.471885, -93.471195, -93.470493, -93.469775, -93.468923, -93.468127, np.nan],
            [np.nan, -93.471324, -93.470554, -93.469704, -93.46888, -93.468096, np.nan],
            [-93.472103, -93.471159, -93.470523, -93.469699, -93.468862, -93.468105, -93.467295],
            [np.nan, np.nan, -93.470504, -93.469708, -93.46887, -93.468061, -93.46725],
            [np.nan, np.nan, np.nan, -93.469716, -93.468933, -93.468055, -93.467377],
            [np.nan, np.nan, np.nan, np.nan, -93.468927, -93.468183, np.nan]]

zones_s2 = [['Upland', 'Upland', 'Upland', 'Upland', 'Upland', 'Upland', np.nan],
            [np.nan, 'Lagg', 'Lagg', 'Bog', 'Upland', 'Upland', np.nan],   
            ['Upland', 'Bog', 'Bog', 'Bog', 'Lagg', 'Lagg', 'Upland'],
            [np.nan, np.nan, 'Lagg', 'Bog', 'Bog', 'Lagg', 'Upland'],
            [np.nan, np.nan, np.nan, 'Upland', 'Upland', 'Upland', 'Upland'],
            [np.nan, np.nan, np.nan, np.nan, 'Upland', 'Upland', np.nan]]

aspect_s2 = [[193.3626, 115.0966, 352.4059, 135.0012, 254.4066, 355.2919, np.nan],
            [np.nan, 332.3497, 291.1621, 120.0787, 9.296235, 201.5048, np.nan],
            [252.4753, 200.2377, 279.4718, 174.7998, 187.7631, 58.388, 275.0663],
            [np.nan, np.nan, 25.94226, 113.9634, 180, 105.2713, 4.635716],
            [np.nan, np.nan, np.nan, 50.67678, 355.6789, 21.695, 308.4806],
            [np.nan, np.nan, np.nan, np.nan, 225, 0.008743, np.nan]]

slope_s2 = [[6.632267, 3.120716, 2.16637, 7.45287, 9.503919, 3.051387, np.nan],
            [np.nan, 0.84887, 2.37949, 0.78623, 3.98498, 1.26977, np.nan],
            [2.85169, 0.72501, 0.43563, 0.39548, 3.17706, 1.09319, 2.38778],
            [np.nan, np.nan, 14.422, 0.70519, 0.28632, 0.40827, 1.32895],
            [np.nan, np.nan, np.nan, 8.63669, 8.02736, 3.57951, 5.336709],
            [np.nan, np.nan, np.nan, np.nan, 0.96229, 1.43197, np.nan]]

'''Build Dataset'''

snow_s6 = xr.Dataset(
    data_vars = {"stakes": (["northing", "easting"], stakes_s6),
                 "latitude": (["northing", "easting"], lat_s6),
                 "longitude": (["northing", "easting"], lon_s6),
                 "zones": (["northing", "easting"], zones_s6),
                 "aspect": (["northing", "easting"], aspect_s6),
                 "slope": (["northing", "easting"], slope_s6),
                 "depths": (["time", "northing", "easting"], data_s6),
                 
    },
    coords={
        "time": times,
        "northing": Northing_s6,
        "easting": Easting_s6
    }
)

snow_s2 = xr.Dataset(
    data_vars = {"stakes": (["northing", "easting"], stakes_s2),
                 "latitude": (["northing", "easting"], lat_s2),
                 "longitude": (["northing", "easting"], lon_s2),
                 "zones": (["northing", "easting"], zones_s2),
                 "aspect": (["northing", "easting"], aspect_s2),
                 "slope": (["northing", "easting"], slope_s2),
                 "depths": (["time", "northing", "easting"], data_s2),
                 
    },
    coords={
        "time": times,
        "northing": Northing_s2,
        "easting": Easting_s2
    }
)


#%%
'''Plotting'''
save_path = '../Figures/snowPlots/'

#Individual depth plots at each time
for t in times:
    snow_s6.depths.sel(time = t).plot.contourf(vmin = 0, vmax = float(snow_s6.depths.max()))
    plt.savefig(save_path + "S6_snowdepths_" + str(t) + ".pdf")
    plt.savefig(save_path + "S6_snowdepths_" + str(t) + ".jpg")
    plt.show()

    snow_s2.depths.sel(time = t).plot.contourf(vmin = 0, vmax = float(snow_s2.depths.max()))
    plt.savefig(save_path + "S2_snowdepths_" + str(t) + ".pdf")
    plt.savefig(save_path + "S2_snowdepths_" + str(t) + ".jpg")
    plt.show()
    
#Depth plots wrapped by time
snow_s6.depths.plot.contourf(col = 'time', col_wrap = 3)
plt.savefig(save_path + "S6_snowdepths_time.pdf")
plt.savefig(save_path + "S6_snowdepths_time.jpg")
plt.show()

snow_s2.depths.plot.contourf(col = 'time', col_wrap = 3)
plt.savefig(save_path + "S2_snowdepths_time.pdf")
plt.savefig(save_path + "S2_snowdepths_time.jpg")
plt.show()


#%%
'''SAVE DATA'''
data_savepath = './Cleaned Data/'

snow_s2.to_netcdf(data_savepath + '01_cleanedsnowdataS2.nc')
snow_s6.to_netcdf(data_savepath + '01_cleanedsnowdataS6.nc')


#%%
'''FUN LITTLE ANIMATION'''
import imageio

with imageio.get_writer(save_path + 's2snow.gif', mode = 'i', fps = 4) as writer:
    for i in times:
        image = imageio.imread(save_path + f'S2_snowdepths_{i}.jpg')
        writer.append_data(image)


with imageio.get_writer(save_path + 's6snow.gif', mode = 'i', fps = 4) as writer:
    for i in times:
        image = imageio.imread(save_path + f'S6_snowdepths_{i}.jpg')
        writer.append_data(image)
# %%
