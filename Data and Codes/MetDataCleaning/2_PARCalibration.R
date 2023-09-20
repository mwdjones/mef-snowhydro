############################################################
# 2_PARcalibration_BLF.R
# jone3247@umn.edu, nina.lany@usda.gov
# created 6/29/2021
# modified 6/23/2021
#
### MUST HAVE 1_make_MetStation_ARCHIVE.R PRELOADED -- Accesses big_BLF
##############################################################

# Check for and install required packages
for (package in c('tidyverse','lubridate', 'ggplot2')) {
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
                  skip=4,
                  na.strings="NAN",
                  stringsAsFactors=F)
  
  # Read in just the header line (l2)
  # unlist the line, and remove quotes 
  h <- readLines(file, n=2)[2]
  n <- as.factor(unlist(strsplit(h, ",")) )
  n2 <- gsub('"', "", n)
  
  # assign column names to dataframe
  colnames(dat) = n2
  
  # catch any excess rows (for NADP Data the SLR columns)
  dat[, c('SlrW_Avg', 'SlrkJ_Tot', 'PAR_Den2_Avg', 'PAR_Tot2_Tot')] = list(NULL)
  
  return(dat)
}

#set date format
tmpDateFormat<-"%Y-%m-%d %H:%M:%S"

# set working directory so it works on both windows and unix OS:
path <- "~/../Box/External-MEF_DATA"
if( .Platform$OS.type == "unix" )
  path <- "~/Box/External-MEF_DATA"

setwd(path)

###SET BASELINE DATA###
base_PAR = data.frame("TIMESTAMP" = big_BLF$TIMESTAMP,
                      "PAR_Den_Avg" = big_BLF$PAR_Den_Avg, 
                      "PAR_Tot_Tot" = big_BLF$PAR_Tot_Tot)

###IMPORT CALIBRATION DATA###
file_list <- list.files("DataDump/BLF/2017", pattern = "Table1.......\\.dat$|Table1_...._.._.._.._.._..\\.dat$|Table1.........\\.dat$|Table1........\\.dat")
weekly_downloads <- list()

for (j in seq_along(file_list)){
  dat <- readCDL(paste0("DataDump/BLF/2017/", file_list[j]))
  
  #Remove the empty column with PA616 and replace with a string of Nans
  if("PA616" %in% colnames(dat)){
    dat = rename(dat, "VW_Avg" = PA616)
    dat$VW_Avg = NA
  }
  
  weekly_downloads[[j]] <- dat
  
}

calib = list()
c = 1
for(i in 1:length(weekly_downloads)){
  if(length(weekly_downloads[[i]]) == 24){
    calib[[c]] = weekly_downloads[[i]]
    c = c + 1
  }
}

calibration <- do.call(rbind, calib) 

#Format DateTime: -- tmpDateFormat defined above, timezone set to CST (America/Chicago)
calibration$TIMESTAMP <- as.POSIXct(calibration$TIMESTAMP, format = tmpDateFormat, tz = "America/Chicago") 
#create rows for missing values: 
datetimes <- seq(min(calibration$TIMESTAMP, na.rm = T), max(calibration$TIMESTAMP, na.rm = T), by = "30 min")
DT <- as.data.frame(datetimes)
calibration <- merge(calibration, DT, by.x = "TIMESTAMP", by.y = "datetimes", all.x = T, all.y = T)


###TECHNIQUE 1: LINEAR REGRESSION - DOY Based###
##PRELIMINARY PLOT##
plot(x = calibration$PAR_Den_Avg, calibration$PAR_Den_new_Avg, 
     xlim = c(0, 2000), ylim = c(0, 2000))
abline(lm(PAR_Den_new_Avg ~ PAR_Den_Avg, data = calibration), col = 'red')
lines(x = c(1:2000), y = c(1:2000), col = "green")

#Relationship between calibration and measured
reg1 = lm(PAR_Den_new_Avg ~ PAR_Den_Avg, data = calibration)
rate = as.double(reg1$coefficients[2])

calibration["PAR_Den_Avg_modelled"] = rate*(calibration$PAR_Den_Avg)

##Follow-up PLOT##
plot(x = calibration$PAR_Den_Avg_modelled, calibration$PAR_Den_new_Avg, 
     xlim = c(0, 2000), ylim = c(0, 2000))
abline(lm(PAR_Den_new_Avg ~ PAR_Den_Avg_modelled, data = calibration), col = 'red')
lines(x = c(1:2000), y = c(1:2000), col = "green")

#Apply to the rest of the dataset
#Sample years -- 2018, 2019

calib2018 = base_PAR[year(base_PAR$TIMESTAMP) == 2018]



