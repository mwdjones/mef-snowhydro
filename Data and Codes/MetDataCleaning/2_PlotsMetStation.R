library(plotly)

xl = as.POSIXct(c('2022-09-01 00:00:00', '2023-07-31 00:00:00'))

#Air Temperature Plots
plot(x = big_South$TIMESTAMP, y = big_South$AirT, type = 'l', 
     xlab = "Time", ylab = "Air Temperature [C]",
     xlim = xl,
     col = alpha('black', 0.5))
lines(x = big_NADP$TIMESTAMP, y = big_NADP$Air_TempC, type = 'l', 
     col = alpha('red', 0.5))
lines(x = big_S2F$TIMESTAMP, y = big_S2F$Air_TempC_Avg,type = 'l', 
     col = alpha('blue', 0.5))
legend('bottomright', legend = c("South", "NADP", "S2 Forest"), 
       col = c('black', 'red', 'blue'), 
       lty=1,lwd=2)

#Relative Humidity Plots=
plot(x = big_South$TIMESTAMP, y = big_South$RH, type = 'l', 
     xlab = "Time", ylab = "RH", 
     xlim = xl,
     col = alpha('black', 0.5))
lines(x = big_NADP$TIMESTAMP, y = big_NADP$RH, type = 'l', 
     col = alpha('red', 0.5))
lines(x = big_S2F$TIMESTAMP, y = big_S2F$RH, type = 'l', 
     col = alpha('blue', 0.5))
legend('bottomright', legend = c("South", "NADP", "S2 Forest"), 
       col = c('black', 'red', 'blue'), 
       lty=1,lwd=2)

#Wind Speed
plot(x = big_NADP$TIMESTAMP, y = big_NADP$WindSpd_S_WVT, type = 'l', 
     col = alpha('red', 0.5), 
     xlim = xl,
     xlab = "Time", ylab = "Wind Speed [m/s]")
lines(x = big_S2F$TIMESTAMP, y = big_S2F$WindSpd_S_WVT, type = 'l', 
     col = alpha('blue', 0.5))
legend('topright', legend = c("NADP", "S2 Forest"), 
       col = c('red', 'blue'), 
       lty=1,lwd=2)

#Wind Direction
plot(x = big_NADP$TIMESTAMP, y = big_NADP$WindDir_D1_WVT, type = 'l', 
     col = alpha('red', 0.5), 
     xlim = xl,
     xlab = "Time", ylab = "Wind Direction [deg]")
lines(x = big_S2F$TIMESTAMP, y = big_S2F$WindDir_D1_WVT, type = 'l', 
      col = alpha('blue', 0.5))
legend('bottomright', legend = c("NADP", "S2 Forest"), 
       col = c('red', 'blue'), 
       lty=1,lwd=2)

#PAR Density
plot(x = big_NADP$TIMESTAMP, y = big_NADP$PAR_Den_Avg, type = 'l', 
     col = alpha('red', 0.5), 
     xlim = xl,
     xlab = "Time", ylab = "PAR Density")
lines(x = big_S2F$TIMESTAMP, y = big_S2F$PAR_Den_Avg, type = 'l', 
      col = alpha('blue', 0.5))
legend('bottomright', legend = c("NADP", "S2 Forest"), 
       col = c('red', 'blue'), 
       lty=1,lwd=2)


#Soil Water Content
plot(x = big_NADP$TIMESTAMP, y = big_NADP$Soil_VWC, type = 'l', 
     col = alpha('red', 0.5), 
     xlim = xl,
     xlab = "Time", ylab = "Soil Water Content")
lines(x = big_S2F$TIMESTAMP, y = big_S2F$Soil_VWC_Avg, type = 'l', 
      col = alpha('blue', 0.5))
legend('bottomright', legend = c("NADP", "S2 Forest"), 
       col = c('red', 'blue'), 
       lty=1,lwd=2)

#Soil Temperature
plot(x = big_NADP$TIMESTAMP, y = big_NADP$Soil_TempC, type = 'l', 
     col = alpha('red', 0.5), 
     xlim = xl,
     xlab = "Time", ylab = "Soil Temperature [C]")
lines(x = big_S2F$TIMESTAMP, y = big_S2F$Soil_TempC_Avg, type = 'l', 
      col = alpha('blue', 0.5))
legend('bottomright', legend = c("NADP", "S2 Forest"), 
       col = c('red', 'blue'), 
       lty=1,lwd=2)




