#%%
'''Imports'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

'''Data'''
#Add data for each sampling date
data_s6 = [[[np.nan, 5.5, 8, np.nan, np.nan, np.nan],
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
            [np.nan, np.nan, np.nan, 22, 32, 23]]]

data_s2 = [[[5, 10, 0, 8.5, 0, 3, np.nan],
            [np.nan, 11, 5, 19, np.nan, 11, np.nan],
            [14.5, 5, 14, 14.5, 5, 8, 8.5],
            [np.nan, np.nan, 9, 10, 9.5, 14, 11],
            [np.nan, np.nan, np.nan, 5.5, 8, 9, 4],
            [np.nan, np.nan, np.nan, np.nan, 12, 5, np.nan]], 
            [[np.nan, 24, 24, 32, 23.5, 25, np.nan],
            [np.nan, 26, 22.5, np.nan, 32, 44, np.nan],
            [47, np.nan, 29, 34, 31, 34, 35],
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
            [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], 
            [[43, 39, 27, 32, 25, 38, np.nan],
            [np.nan, 41, 36, 31, 38, 48, np.nan],
            [41, 31, 43, 46, 39, 56, 40],
            [np.nan, np.nan, 41, 45, 41, 42, 43],
            [np.nan, np.nan, np.nan, 43, 46, 44, 44],
            [np.nan, np.nan, np.nan, np.nan, 39, 38, np.nan]]]

'''Times'''
#Add dates as needed
times = np.array(['12-02-2022', '12-30-2022', '01-05-2023', '01-13-2023', '01-20-2023', '02-01-2023', '02-10-2023'])

'''Dims'''
#These are averaged from the stake coordinates to make the data fit a grid system
Northing_s6 = np.array(np.linspace(464644, 464335, 6))
Easting_s6 = np.array(np.linspace(5262240, 5263285, 6))

stakes_s6 = [[np.nan, 'S655', 'S654', np.nan, np.nan, np.nan],
            ['S646', 'S645', 'S644', 'S643', 'S642', np.nan],
            ['S636', 'S635', 'S634', 'S633', 'S632', 'S631'],
            [np.nan, 'S625', 'S624', 'S623', 'S622', 'S621'],
            [np.nan, np.nan, 'S614', 'S613', 'S612', 'S611'],
            [np.nan, np.nan, np.nan, 'S603', 'S602', 'S601']]

zones_s6 = [[np.nan, 'Upland', 'Upland', np.nan, np.nan, np.nan],
            ['Upland', 'Bog', 'Lagg', 'Upland', 'Upland', np.nan], #s644 should be lagg - diff from Kristina LAI analysis
            ['Upland', 'Lagg', 'Bog', 'Lagg', 'Upland', 'Upland'],
            [np.nan, 'Upland', 'Lagg', 'Bog', 'Lagg', 'Lagg'], #s621 should be lagg - diff from Kristina LAI analysis
            [np.nan, np.nan, 'Upland', 'Lagg', 'Bog', 'Lagg'],
            [np.nan, np.nan, np.nan, 'Upland', 'Upland', 'Upland']]

Northing_s2 = np.array(np.linspace(464819, 464448, 6)) 
Easting_s2 = np.array(np.linspace(5262240, 5262552, 7))

stakes_s2 = [['S200', 'S201', 'S202', 'S203', 'S204', 'S205', np.nan],
            [np.nan, 'S211', 'S212', 'S213', 'S214', 'S215', np.nan],
            ['S220', 'S221', 'S222', 'S223', 'S224', 'S225', 'S226'],
            [np.nan, np.nan, 'S232', 'S233', 'S234', 'S235', 'S236'],
            [np.nan, np.nan, np.nan, 'S243', 'S244', 'S245', 'S246'],
            [np.nan, np.nan, np.nan, np.nan, 'S254', 'S255', np.nan]]

zones_s2 = [['Upland', 'Upland', 'Upland', 'Upland', 'Upland', 'Upland', np.nan],
            [np.nan, 'Lagg', 'Lagg', 'Bog', 'Upland', 'Upland', np.nan],   
            ['Upland', 'Bog', 'Bog', 'Bog', 'Lagg', 'Lagg', 'Upland'],
            [np.nan, np.nan, 'Lagg', 'Bog', 'Bog', 'Lagg', 'Upland'],
            [np.nan, np.nan, np.nan, 'Upland', 'Upland', 'Upland', 'Upland'],
            [np.nan, np.nan, np.nan, np.nan, 'Upland', 'Upland', np.nan]]

'''Build Dataset'''

snow_s6 = xr.Dataset(
    data_vars = {"stakes": (["northing", "easting"], stakes_s6),
                 "zones": (["norting", "easting"], zones_s6),
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
                 "zones": (["northing", "easting"], zones_s2),
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
save_path = 'D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Figures/snowPlots/'

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

data_savepath = 'D:/1_DesktopBackup/Feng Research/0_MEF Snow Hydology/mef-snowhydro/Data and Codes/Cleaned Data/'

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
