# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:32:47 2024

@author: jonyegbula
"""


import os
import ee
import pandas as pd

mydir = "Code"      # adjust directory
os.chdir(mydir)

# read csv file and convert dates from strings to datetime
data = pd.read_csv("csv/meadow.csv")
data.head()

# Authenticate and Initialize the Earth Engine API
# ee.Authenticate()
ee.Initialize()

# reads Landsat data, flow accumulation, gridmet temperature and DEM data (for slope and elevation)
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
dem = ee.Image('USGS/3DEP/10m').select('elevation')
slopeDem = ee.Terrain.slope(dem)

def maskClouds(image):
    # mask out cloud based on bits in QA_pixel
    qa = image.select('QA_PIXEL')
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    # update image with combined mask
    cloud_mask = dilated_cloud.And(cloud).And(cloudShadow)
    return image.updateMask(cloud_mask)

Blue, Green, Red, NIR, SWIR_1, SWIR_2 = [], [], [], [], [], []
flow, slope, elevation = [], [], []
min_summer, max_summer, min_winter, max_winter = [], [], [], []
# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    year = target_date[:4]
    
    # filter landsat by location and year, and sort by NIR (B5) then extract band values
    spatial_filtered = landsat8_collection.filterBounds(point).map(maskClouds).first()
    bands = spatial_filtered.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    band_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    band_values = list(band_values.values())
    mycrs = spatial_filtered.projection()
    
    # compute min and max summer and winter temperature from gridmet (resolution = 4,638.3m)
    gridmet_tmmn = gridmet.filterBounds(point).filterDate(str(int(year)-1)+'-10-01', year+'-03-31').select('tmmn')
    gridmet_tmmx = gridmet.filterBounds(point).filterDate(year+'-04-01', year+'-09-30').select('tmmx')
    min_winter.append(gridmet_tmmn.min().reduceRegion(ee.Reducer.min(), point, 30).getInfo()['tmmn'])
    max_winter.append(gridmet_tmmn.max().reduceRegion(ee.Reducer.max(), point, 30).getInfo()['tmmn'])
    min_summer.append(gridmet_tmmx.min().reduceRegion(ee.Reducer.min(), point, 30).getInfo()['tmmx'])
    max_summer.append(gridmet_tmmx.max().reduceRegion(ee.Reducer.max(), point, 30).getInfo()['tmmx'])
    
    # compute flow accumulation (463.83m resolution); slope and aspect (10.2m resolution); daymetv4 (1km resolution)
    flow_30m = flow_acc.resample('bilinear').reproject(crs=mycrs, scale=30)
    dem_30m = dem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear').reproject(crs=mycrs, scale=30)
    slope_30m = slopeDem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear').reproject(crs=mycrs, scale=30)
    flow_value = flow_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    elev = dem_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
    slope_value = slope_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
    
    Blue.append(band_values[0])
    Green.append(band_values[1])
    Red.append(band_values[2])
    NIR.append(band_values[3])
    SWIR_1.append(band_values[4])
    SWIR_2.append(band_values[5])
    flow.append(flow_value)
    elevation.append(elev)
    slope.append(slope_value)
    
    if idx%50 == 0: print(idx, end=' ')

data['Blue'] = [(x*2.75e-05 - 0.2) for x in Blue]
data['Green'] = [(x*2.75e-05 - 0.2) for x in Green]
data['Red'] = [(x*2.75e-05 - 0.2) for x in Red]
data['NIR'] = [(x*2.75e-05 - 0.2) for x in NIR]
data['SWIR_1'] = [(x*2.75e-05 - 0.2) for x in SWIR_1]
data['SWIR_2'] = [(x*2.75e-05 - 0.2) for x in SWIR_2]

data['Flow'] = flow
data['Elevation'] = elevation
data['Slope'] = slope
data['Min_summer_temp'] = min_summer
data['Max_summer_temp'] = max_summer
data['Min_winter_temp'] = min_winter
data['Max_winter_temp'] = max_winter

data['NDVI'] = (data['NIR'] - data['Red'])/(data['NIR'] + data['Red'])
data['NDWI'] = (data['Green'] - data['NIR'])/(data['Green'] + data['NIR'])
data['EVI'] = 2.5*(data['NIR'] - data['Red'])/(data['NIR'] + 6*data['Red'] - 7.5*data['Blue'] + 1)
data['SAVI'] = 1.5*(data['NIR'] - data['Red'])/(data['NIR'] + data['Red'] + 0.5)
data['BSI'] = ((data['Red'] + data['SWIR_1']) - (data['NIR'] + data['Red']))/(data['Red'] + data['SWIR_1'] + data['NIR'] + data['Red'])

data.head()

# checks how many pixels are cloud free (non-null value);
ids = [x for x in Blue if x]
len(ids)

# write updated dataframe to new csv file
data.to_csv('csv/meadow_Data.csv', index=False)
