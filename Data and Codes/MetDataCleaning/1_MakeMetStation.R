#############################################################
# 1_make_MetStation_ARCHIVE.R
# jone3247@umn.edu, nina.lany@usda.gov
# created 6/21/2021
# modified 7/7/2021
#
# Purpose: To collect and clean all 30 min Met Station data from the Campbell Met stations. Currently
#   data are contained within .csv files separated by week and year in the Box folder. 
#   This process will collate all data files together and remove any unnecessary information
#   printed by the sensors. 
# Notes: Some of the raw .dat files in the Box Drive have been renamed at the end of the name to exclude
#   them from the algorithm. This is on purpose and helps eliminate obscure files. 
#############################################################

rm(list = ls())

# Check for and install required packages
for (package in c('tidyverse','lubridate', 'ggplot2', 'data.table')) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package)
    library(package, character.only=T)
  }
}

# Some handy functions:
#function for rounding numbers in a vector a to the nearest b (default is 0.5, for HMP sensors)
round_special <- function(a,b =0.5) {
  a = round(a / b) * b
}


#load Campbell datalogger file
readCDL=function(file){
  
  # read data file starting on 5th line
  dat <- read.csv(file,
                  header=FALSE,
                  skip=3,
                  na.strings="NAN",
                  stringsAsFactors=F)
  
  # Read in just the header line (l2)
  # unlist the line, and remove quotes 
  h <- readLines(file, n=2)[1]
  n <- as.factor(unlist(strsplit(h, ",")) )
  n2 <- gsub('"', "", n)
  
  # assign column names to dataframe
  colnames(dat) = n2
  
  # catch any excess rows (for NADP Data the SLR columns)
  dat[, c('SlrW_Avg', 'SlrkJ_Tot', 'PAR_Den2_Avg', 'PAR_Tot2_Tot', 'PAR_Den_new_Avg', 'PAR_Tot_new_Tot')] = list(NULL)
  
  return(dat)
}

testRange = function(data, n, s){
  #Data is the data you are checking for outliers
  #n is the look-back length of numbers you want to compare it to
  #s is the number of standard deviations from the mean that count as outliers
  
  len = length(data)
  outliers = c()
  j = 1
  for(i in seq(n+1, len)){
    #If the value is a NA, skip this loop
    if(is.na(data[i])){
      next
    }
    
    #Set the start and end values to compare against
    start = i-n
    end = i-1
    
    #Take the previous five values
    set = data[start:end]
    av = mean(set, na.rm = TRUE)
    std = sd(set, na.rm = TRUE)
    
    #The selected data point must be within two stdevs, or else it is ruled an outlier
    upper = av+s*(std)
    lower = av-s*(std)
    
    #Error catching
    if((is.na(upper)) | (is.na(lower))){
      next
    }
    
    #If the selected value is outside 2SDEV, consider it an outlier
    if((data[i] > upper) | (data[i] < lower)){
      outliers[j] = i
      j = j + 1
    }
    
  }
  
  return(outliers)
}



#set date format
tmpDateFormat<-"%Y-%m-%d %H:%M:%S"

##############
#    NADP    #
##############

big_NADP <- readCDL('Data and Codes/Raw Data/ATM/NADP_Met_Met.dat')

### CLEANING ###
#Check if timestamps are unique
uniq = length(unique(big_NADP$TIMESTAMP)) == nrow(big_NADP)

#Remove duplicated rows
if(!uniq){
  big_NADP = big_NADP[!duplicated(big_NADP),]
}

#Format DateTime: -- tmpDateFormat defined above, timezone set to CST (America/Chicago)
big_NADP$TIMESTAMP <- as.POSIXct(big_NADP$TIMESTAMP, format = tmpDateFormat, tz = "America/Chicago") 
#create rows for missing values: 
datetimes <- seq(min(big_NADP$TIMESTAMP, na.rm = T), max(big_NADP$TIMESTAMP, na.rm = T), by = "30 min")
DT <- as.data.frame(datetimes)
big_NADP <- merge(big_NADP, DT, by.x = "TIMESTAMP", by.y = "datetimes", all.x = T, all.y = T)

#round air temp column to nearest 0.1
big_NADP$Air_TempC <-round(big_NADP$Air_TempC, 1)

#round RH column to nearest 1%
big_NADP$RH <- round(big_NADP$RH, 0)

#round WindDir to 0.01
big_NADP$WindDir_D1_WVT <-round(big_NADP$WindDir_D1_WVT, 2)
big_NADP$WindDir_SD1_WVT <-round(big_NADP$WindDir_SD1_WVT, 2)

### QAQC ###
#Volumetric Water Content in range of [0, 1]
big_NADP["Soil_VWC"] <- lapply(big_NADP["Soil_VWC"], function(x) replace(x, x < 0 | x > 1, NA))

#PAR Density Average in range of [0, 2000]
big_NADP["PAR_Den_Avg"] <- lapply(big_NADP["PAR_Den_Avg"], function(x) replace(x, x < 0 | x > 2000, NA))

#Wind Speed in range of [0, 5]
big_NADP["WindSpd_S_WVT"] <- lapply(big_NADP["WindSpd_S_WVT"], function(x) replace(x, x < 0 | x > 5, NA))

#Relative Humidity should be on the range [0, 100] and then remove the strange data around 2014
big_NADP["RH"] <- lapply(big_NADP["RH"], function(x) replace(x, x < 0 | x > 100, NA))

#Soil Temperature
big_NADP["Soil_TempC"] <- lapply(big_NADP["Soil_TempC"], function(x) replace(x, x < -5 | x > 30, NA))
## Remove NADP SoilT from 2019-Present
#big_NADP$Soil_TempC_Avg[big_NADP$TIMESTAMP > '2019-01-01 00:00:00'] <- NA

#Air Temperature
#big_NADP$Air_TempC_Avg[(big_NADP$TIMESTAMP > '2015-11-01 00:00:00') & (big_NADP$TIMESTAMP < '2016-11-01 01:00:00')] <- NA


summary(big_NADP)
str(big_NADP)


##############
#   South    #
##############


big_South <- readCDL('Data and Codes/Raw Data/ATM/South_Wx_SouthMet.dat')


##CODE FOR PICKING OUT UNMATCHED NAMES
#nameTemp = colnames(weekly_downloads[[1]])
#for(i in 1:length(weekly_downloads)){
#  new = colnames(weekly_downloads[[i]])
#  if(!identical(new, nameTemp)){
#    print("-----")
#    print(i)
#    print(file_list[i])
#  }}

### CLEANING ###
#Check if timestamps are unique
uniq = length(unique(big_South$TIMESTAMP)) == nrow(big_South)

#Remove duplicated rows
if(!uniq){
  big_South = big_South[!duplicated(big_South),]
}

#Format DateTime: -- tmpDateFormat defined above, timezone set to CST (America/Chicago)
big_South$TIMESTAMP <- as.POSIXct(big_South$TIMESTAMP, format = tmpDateFormat, tz = "America/Chicago")
#create rows for missing values:
datetimes <- seq(min(big_South$TIMESTAMP, na.rm = T), max(big_South$TIMESTAMP, na.rm = T), by = "30 min")
DT <- as.data.frame(datetimes)
big_South <- merge(big_South, DT, by.x = "TIMESTAMP", by.y = "datetimes", all.x = T, all.y = T)

#round air temp column to nearest 0.1
big_South$AirT <-round(big_South$AirT, 1)

#round RH column to nearest 1%
big_South$RH <- round(big_South$RH, 0)

#remove timestamps before 2020
big_South <- big_South[big_South$TIMESTAMP > '2020-01-01 00:00:00', ] 


## QAQC ###
#Relative Humidity should be on the range [0, 100]
big_South["RH"] <- lapply(big_South["RH"], function(x) replace(x, x < 0 | x > 100, NA))

#Air Temperature Missing Values
big_South["AirT"] <- lapply(big_South["AirT"], function(x) replace(x, x < -50 | x > 50, NA))
##Air Temperature Remove outliers
out <- testRange(big_South$AirT, 2000, 4)
big_South$AirT[out] <- NA

summary(big_South)
str(big_South)

###################
#    S2 Forest    #
###################


big_S2F <- readCDL('Data and Codes/Raw Data/ATM/S2_forest_met_Table1.dat')

### CLEANING ###
#Check if timestamps are unique
uniq = length(unique(big_S2F$TIMESTAMP)) == nrow(big_S2F)

#Remove duplicated rows
if(!uniq){
  big_S2F = big_S2F[!duplicated(big_S2F),]
}

#Format DateTime: -- tmpDateFormat defined above, timezone set to CST (America/Chicago)
big_S2F$TIMESTAMP <- as.POSIXct(big_S2F$TIMESTAMP, format = tmpDateFormat, tz = "America/Chicago")
#create rows for missing values:
datetimes <- seq(min(big_S2F$TIMESTAMP, na.rm = T), max(big_S2F$TIMESTAMP, na.rm = T), by = "30 min")
DT <- as.data.frame(datetimes)
big_S2F <- merge(big_S2F, DT, by.x = "TIMESTAMP", by.y = "datetimes", all.x = T, all.y = T)

#round air temp column to nearest 0.1
big_S2F$Air_TempC_Avg <-round(big_S2F$Air_TempC_Avg, 1)

#round RH column to nearest 1%
big_S2F$RH <- round(big_S2F$RH, 0)

#round WindDir to 0.01
big_S2F$WindDir_D1_WVT <-round(big_S2F$WindDir_D1_WVT, 2)
big_S2F$WindDir_SD1_WVT <-round(big_S2F$WindDir_SD1_WVT, 2)


### QAQC ###
#Volumetric Water Content in range of [0, 1]
big_S2F["Soil_VWC_Avg"] <- lapply(big_S2F["Soil_VWC_Avg"], function(x) replace(x, x < 0 | x > 1, NA))

#PAR Density Average in range of [0, 2000]
big_S2F["PAR_Den_Avg"] <- lapply(big_S2F["PAR_Den_Avg"], function(x) replace(x, x < 0 | x > 2000, NA))

#Wind Speed in range of [0, 5]
big_S2F["WindSpd_S_WVT"] <- lapply(big_S2F["WindSpd_S_WVT"], function(x) replace(x, x < 0 | x > 5, NA))
## Remove pre-2008
#big_S2F$WindSpd_S_WVT[big_S2F$TIMESTAMP < '2008-01-01 00:00:00'] <- NA

#Precipitation set to range [0, 1] and get rid of all days after the
#massive june event 2015-06-23 13:00:00
big_S2F["Rain_inch_Tot"] <- lapply(big_S2F["Rain_inch_Tot"], function(x) replace(x, x < 0 | x > 1, NA))
#big_S2F$Rain_inch_Tot[big_S2F$TIMESTAMP > '2015-06-23 13:00:00'] <- NA

#Relative Humidity should be on the range [0, 100] and then remove the strange data around 2014
big_S2F["RH"] <- lapply(big_S2F["RH"], function(x) replace(x, x < 0 | x > 100, NA))
#big_S2F$RH[(big_S2F$TIMESTAMP > '2013-07-09 15:00:00') & (big_S2F$TIMESTAMP < '2014-09-25 10:30:00')] <- NA

summary(big_S2F)
str(big_S2F)

####
# Export Cleaned Data
write.csv(big_S2F, 'Data and Codes/Cleaned Data/ATM/01_cleanedS2F.csv')
write.csv(big_South, 'Data and Codes/Cleaned Data/ATM/01_cleanedSouthMet.csv')
write.csv(big_NADP, 'Data and Codes/Cleaned Data/ATM/01_cleanedNADP.csv')