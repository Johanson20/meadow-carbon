library(gstat)
library(sp)
library(rgl)

graphics.off()
rm(list = ls())

setwd("C:/Users/jonyegbula/Documents/PointBlue/Code")
data <- read.csv("csv/data.csv")
ids = which(is.na(data$HerbBio.g.m2) == T)
data <- data[-ids, ]
coordinates <- cbind(data$x, data$y)
spdf <- SpatialPointsDataFrame(coordinates, data)
spdf$SampleDate <- as.Date(spdf$SampleDate, format="%m/%d/%Y")
plot(HerbBio.g.m2 ~ x+y, spdf)

vg <- variogram(HerbBio.g.m2 ~ 1, data=spdf, width=100)
plot(vg, pch=16)
v = fit.variogram(vg, vgm(250000, "Gau", 87000, 10000))
v
plot(vg, model=v, pch=21, col='red', cex=0.5, xlim = c(0, 180000), ylim = c(0, 700000))

data2015 <- subset(spdf, (SampleDate >= '2015-01-01') & (SampleDate <= '2015-12-31'))
dim(data2015)

plot3d(data2015$x, data2015$y, data2015$HerbBio.g.m2, col='blue')
rglwidget()
vg <- variogram(HerbBio.g.m2 ~ 1, data=data2015, width=10)
plot(vg, pch=16)
v = fit.variogram(vg, vgm(400, "Sph", 200, 0)) # psill,model,range,nugget
v
plot(vg, model=v, pch=21, col='red', cex=0.5)
