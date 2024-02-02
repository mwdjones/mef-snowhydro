import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
import scipy
import xarray as xr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import datetime as dt

from scipy.signal import find_peaks
from scipy.stats import kendalltau

from math import floor, log10

def findFirstNonZero(list):
    for index, value in enumerate(list):
        if(value > 0):
                return value
        else:
            continue    
    return np.nan

def findFirstNonZero_Reverse(list):
    for index, value in reversed(enumerate(list)):
        if(value > 0):
                return value
        else:
            continue    
    return np.nan

def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)


def jitter(values, j):
    return values + np.random.normal(j, 0.1, values.shape)

def positive_cumsum(x):
    y = np.zeros(len(x))

    for i in range(1, len(x)-1):
        if(y[i-1] + x[i-1] < 0):
            y[i] = 0
        else:
            y[i] = y[i-1] + x[i-1]

    return y

def modelSWE(stake, lai_df, snow_df, precip):
    #Simple numerical model for snow water equivalent from snow inputs using LAI to calculate interception and near suface
    #air temperature to calculate melt potential
    #Inputs: stake - string referencing the site location in lai_df
    #        lai_df - dataframe containing lead area index values for all sites in the winter
    #        snow_df - dataframe containing accumulated snow depths by site (for comparison)
    #        precip - dataframe containing snowfall inputs and air temp (site neutral)
    #lai at the site
    lai = float(lai_df['OLS Prediction Ring 5'].loc[lai_df.Stake_ID == stake].values)

    #select relevant data
    temp = snow_df[snow_df.stakes == stake].reset_index(drop = True)

    #merge with snow data
    temp2 = pd.merge(temp, precip[['Date', 'Tmax_F', 'Tmin_F', 'Snow_mm']], right_on = 'Date', left_on = 'time', how = 'right')

    #Temperature conversions
    temp2['Tavg_F'] = (temp2.Tmax_F + temp2.Tmin_F)/2
    temp2['Tavg_C'] = (temp2.Tavg_F - 32)/1.8000

    #Estimate Accumulation -- done in mm
    temp2['LAI'] = lai
    temp2['CanopyDensity'] = 0.29*(1.9 + np.log(temp2.LAI))
    #temp2['Snow_mm_underCanopy'] = temp2.Snow_mm * [1 - ((0.144*np.log(lai))+0.223)]
    temp2['Snow_mm_underCanopy'] = temp2.Snow_mm * [1 - ((0.144*lai)+0.223)] #test non log transformed interception rate
    temp2['Interception_mm'] = temp2.Snow_mm - temp2.Snow_mm_underCanopy

    #Estimate melt -- done in cm 
    temp2['DDF'] = 0.292-0.164*temp2.CanopyDensity
    temp2['Melt_cm_potential'] = np.where(((temp2.Tavg_C > 0) & (temp2.Snow_mm == 0)), temp2.DDF*(temp2.Tavg_C - 0), 0)

    temp2['Snow_cm_underCanopy'] = temp2['Snow_mm_underCanopy']/10
    temp2['deltaSWE'] = temp2.Snow_cm_underCanopy - temp2.Melt_cm_potential
    temp2['SWE_cm'] = positive_cumsum(temp2.deltaSWE)
    temp2['Melt_cm_modelled'] = np.where(np.diff(temp2.SWE_cm, append = 0) < 0, -np.diff(temp2.SWE_cm, append = 0), 0)

    #reset stake label and site characteristics
    temp2['stakes'] = stake
    temp2['northing'] = temp.northing[0]
    temp2['easting'] = temp.easting[0]
    temp2['zones'] = temp.zones[0]
    temp2['aspect'] = temp.aspect[0]
    temp2['slope'] = temp.slope[0]
    temp2['watershed'] = temp.watershed[0]

    return temp2

def spikeDetect(dat, var, windowBegin, windowEnd):
    #Description - This function takes in the entire vector of water table levels, soil moisture levels, etc. and returns the best 
    #   fit beginning and end to the snow melt signal
    #Packages - numpy (np), pandas (pd), seaborn (sns), matplotlib.pyplot (plt), 
    #   scipy.signal (find_peaks), scipy.stats (kendalltau) 
    #Input - dat: an mxn pandas dataframe containing at least three columns: 
    #               The variable of interest 'var'
    #               "DOY" containing the day of the year 
    #               "YEAR" containing the year of the data point
    #        var: the variable of interest in the dataframe 
    #        windowBegin: the date of the year of the begining of the spring season
    #        windowEnd: the date of the year of the end of the spring season
    #Outputs - WTETri: 6xN dataframe with the following format ['YEAR', WTE', 'Peak', 'Trough', 
    #                                                           'LengthClimb', 'HeightClimb']
    #        - One plot as detailed below
    #        - text ouputs detailing the peaks, troughs and their correlations
    
    Tri = dat[[var, 'YEAR']].groupby('YEAR', as_index = False).mean()
    
    Tri["Peak"] = list(range(0, len(Tri.YEAR)))
    Tri["Trough"] = list(range(0, len(Tri.YEAR)))
    Tri["LengthClimb"] = list(range(0, len(Tri.YEAR)))
    Tri["HeightClimb"] = list(range(0, len(Tri.YEAR)))
    
    Tri["PeakVar"] = list(range(0, len(Tri.YEAR)))
    Tri["TroughVar"] = list(range(0, len(Tri.YEAR)))
    
    for i in Tri.YEAR:
        
        print("-----\n")
         
        #Subset the data for the year of interest
        year = dat[var][dat.YEAR == i].reset_index(drop = True)
            
        print("Year: " + str(i) + "\n")

        #Subset the timeseries to the period of interest
        season = dat[var][(dat.DOY > windowBegin) & (dat.DOY < windowEnd)]
        
        #Calculated the percentiles for limiting peaks and trough selection        
        per5 = np.percentile(season, 10)
        per95 = np.percentile(season, 90)
        
        print("Seasonal Statistics")
        print("95th Percentile: " + str(per95))
        print("50th Percentile: " + str(np.mean(season)))
        print("5th Percentile: " + str(per5) + "\n")
        
        #Identify peaks during the selected time of the year and above the threshold
        peaks, _ = find_peaks(season, height = per95)
        #peaks = peaks + 30
        
        print("Peaks: " + str(peaks) + "\n")
        
        #Identify troughs during the selected time of year and below the threshold
        troughs, _ = find_peaks(-season, height = -per5)
        #troughs = troughs + 30
        
        print("Throughs: " + str(troughs) + "\n")
        
        #Create the matrix to hold the correlation coefficients
        coefficients = pd.DataFrame(np.zeros((len(troughs), len(peaks))))
        
        #Cycle through the rows (troughs) and columns (peaks)
        for index, row in coefficients.iterrows():
            t = troughs[index]
            
            for j in np.arange(0, len(peaks)):
                p = peaks[j]
                
                #print("Peak: " + str(p) + "   Trough: " + str(t))
                
                #Pull the real data and generate modeled data (straight line)
                real = year[t:p]
                step_size = (year[p]-year[t])/(len(real))
                model = np.arange(year[t], year[p], step_size)
                
                #For some reason a few years end up with unequal lengths
                if len(model) != len(real):
                    continue
                
                #Compare using the correlation coefficient
                coeff, _ = kendalltau(model, real, nan_policy = 'omit')
                
                #print("Kendall Tau: " + str(coeff) + "\n")
        
                row[j] = coeff
            
        #Pull out the highest correlation from the matrix, assign based on indices
        best = coefficients.max().max()
        best_t, best_p = list(np.where(coefficients == best))
        best_peak = peaks[best_p]
        best_trough = troughs[best_t]
        
        #PLOT: Shows the water table plot for that year overlayed with the 
        #   identified peaks and troughs
        fig = plt.gcf()
        fig.set_size_inches(6, 3)
        plt.plot(dat[dat.YEAR == i][var])
        plt.axvspan(min(season.index), max(season.index), color = 'b', alpha = 0.25, lw = 0)
        plt.plot(peaks, season[peaks], 'og')
        plt.plot(troughs, season[troughs], 'or')
        plt.plot(best_trough, season[best_trough], 'oy')
        plt.plot(best_peak, season[best_peak], 'oy')
        plt.title("Daily Change in " + var + ', ' + str(i))
        plt.show()


        #Assign all the values back to the dataframe
        Tri.loc[Tri.YEAR == i, 'Peak'] = best_peak
        Tri.loc[Tri.YEAR == i, 'Trough'] = best_trough
        Tri.loc[Tri.YEAR == i, 'LengthClimb'] = (best_peak - best_trough)
        Tri.loc[Tri.YEAR == i, 'HeightClimb'] = float(year[best_peak]) - float(year[best_trough])
        
        Tri.loc[Tri.YEAR == i, 'PeakVar'] = float(year[best_peak])
        Tri.loc[Tri.YEAR == i, 'TroughVar'] = float(year[best_trough])
        
        print("Best Peak: " + str(best_peak) + "   Best Trough: " + str(best_trough) + "\n")   
    
    return Tri