# clears every plot and existing variable
graphics.off()
rm(list=ls())

library(terra)

mydir <- "C:/Users/jonyegbula/Documents/PointBlue/Code"
setwd(mydir)

# function to plot timeseries of multiband images (provide full path name) or a folder of single band images (provide unzipped folder name)
plotTimeSeries <- function(imagepath, years){
  isFolder <- file.info(imagepath)$isdir
  if (isFolder){ # folder of single band geotiffs
    myraster <- list.files(imagepath, pattern = "\\.tif$", full.names = TRUE)
    noLayers <- length(myraster)
  } else { # single multiband geotiff
    myraster <- rast(imagepath)
    noLayers <- nlyr(myraster)
    # print(names(myraster))
  }
  
  band_means <- rep(0, noLayers)
  band_medians <- rep(0, noLayers)
  band_sds <- rep(0, noLayers)
  
  for (i in 1:noLayers) {
    if (isFolder) {    band <- rast(myraster[[i]])
    } else {
      band <- myraster[[i]]
    }
    # Compute statistics over all cells
    vals <- values(band, na.rm = TRUE)
    band_means[i] <- mean(vals)
    band_medians[i] <- median(vals)
    band_sds[i] <- sd(vals)
  }
  
  # 4️⃣ Create a data frame with resultsand create time series
  time_series <- data.frame(Year = years, Mean = band_means,  Median = band_medians, SD = band_sds)
  # Plot mean first
  plot(time_series$Year, time_series$Mean, type = "o", col = "blue", pch = 16,
       xlab = "Year", ylab = "Value", main = "Time Series of Raster Statistics")
  # Add median
  lines(time_series$Year, time_series$Median, type = "o", col = "green", pch = 17)
  # Add standard deviation
  lines(time_series$Year, time_series$SD, type = "o", col = "red", pch = 18)
  # Add legend
  legend("topright", legend = c("Mean", "Median", "SD"), col = c("blue", "green", "red"), pch = c(16, 17, 18), lty = 1)
}

plotTimeSeries("files/single_band_NEP", c(1986:2024))
plotTimeSeries("files/multi_band_NEP.tif", c(1986:2024))
