# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 15:09:32 2025

@author: Johanson C. Onyegbula (johansononyegbula20@gmail.com)
"""

import os
import numpy as np
import warnings
import pandas as pd
import geopandas as gpd
import ee
from shapely.geometry import box

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)
warnings.filterwarnings("ignore")
ee.Initialize()


def maskAndRename(image):
    # rename bands and mask out cloud based on bits in QA_pixel; then scale values
    image = image.rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'QA'])
    qa = image.select('QA')
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    snow = qa.bitwiseAnd(1 << 5).eq(0)
    cloud_mask = dilated_cloud.And(cirrus).And(cloud).And(cloudShadow)
    cloud_and_snow_mask = cloud_mask.And(snow)
    image = image.updateMask(cloud_and_snow_mask).select(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2'])
    scaled_bands = image.multiply(2.75e-05).add(-0.2)
    return image.addBands(scaled_bands, overwrite=True)


def calculateIndices(image):
    # calculate and add indices from landsat band values, and return only indices
    ndwi = image.normalizedDifference(['NIR', 'SWIR_1']).rename('NDWI')
    evi = image.expression("2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'BLUE': image.select('Blue')}).rename('EVI')
    bsi = image.expression("((RED + SWIR_1) - (NIR + BLUE)) / (RED + SWIR_1 + NIR + BLUE)", {'RED': image.select('Red'), 'SWIR_1': image.select('SWIR_1'), 'NIR': image.select('NIR'), 'BLUE': image.select('Blue')}).rename('BSI')
    image = image.addBands([ndwi, evi, bsi])
    return image.select(["NDWI", "EVI", "BSI"])

# Function to extract cloud free band values per pixel from landsat 8 or landsat 7
def getBandValues(shapefile_bbox, target_date, bufferDays = 30):
    year, month, day = target_date.split("-")
    
    # Calculates absolute time difference (in days) from a target date, in which the images are acquired
    def calculate_time_difference(image):
        time_difference = ee.Number(image.date().difference(target_date, 'day')).abs()
        return image.set('time_difference', time_difference)
    
    try:
        # filter landsat images by location
        spatial_filtered = landsat.filterBounds(shapefile_bbox)
        try:
            nppVal = landsat_npp.filterDate(year+"-01-01", year+"-12-31").first().reduceRegion(ee.Reducer.mean(), shapefile_bbox, 30).getInfo()
        except:
            nppVal = None
        # filter the images by dates on or bufferDays before fire date
        temporal_filtered = spatial_filtered.filterDate(ee.Date(target_date).advance(-bufferDays, 'day'), target_date)
        # Sort the images by closest date to fire date and create an image list to loop through if there are valid images
        sorted_collection = temporal_filtered.map(calculate_time_difference).sort('time_difference')
        image_list = sorted_collection.toList(sorted_collection.size())
        noImages = image_list.size().getInfo()
        nImage, band_values = 0, {'NDWI_mean': None}
        reducerFunc = (ee.Reducer.mean().combine(ee.Reducer.median(), sharedInputs=True).combine(ee.Reducer.stdDev(), sharedInputs=True))
        
        # repeatedly check for cloud free pixels (non-null value) in landsat 8, or checks in landsat 7
        while band_values['NDWI_mean'] == None and nImage < noImages:
            nearest_image = calculateIndices(ee.Image(image_list.get(nImage)))
            nImage += 1
            properties = nearest_image.getInfo()['properties']
            band_values = nearest_image.reduceRegion(reducerFunc, shapefile_bbox, 30).getInfo()
        
        return [list(band_values.values()), nppVal, properties['time_difference'], properties['DATE_ACQUIRED'], properties['SCENE_CENTER_TIME']]
    except:     # if there was error in retrieving values due to no data available within dates
        return []

# read meadow data and extract entire boundary for filtering GEE images
epsg_crs = "EPSG:4326"
meadows = gpd.read_file("files/meadows_dateBeforeFire_fires2012to2023_20251210.shp").to_crs(epsg_crs)
minx, miny, maxx, maxy = meadows.total_bounds
merged_zones = gpd.GeoDataFrame([1], geometry=[box(minx, miny, maxx, maxy)], crs=epsg_crs)
meadow_zone = ee.Geometry.Polygon(list(merged_zones.geometry[0].exterior.coords)).buffer(100)

# load all landsat images (order by most recent landsat) and npp values
landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat = landsat9.merge(landsat8).merge(landsat7).merge(landsat5).filterBounds(meadow_zone).map(maskAndRename)
landsat_npp = ee.ImageCollection("UMT/NTSG/v2/LANDSAT/NPP").select('annualNPP')

# create empty dataframe, one for each meadow
meadow_data = pd.DataFrame(index=np.arange(meadows.shape[0]), columns=['UniqueID', 'FireID', 'Fire_Date',
            'Image_Date', 'Image_Time', 'Days_Difference', 'LandsatNPP', 'BSI_mean', 'BSI_median', 'BSI_std',
            'EVI_mean','EVI_median', 'EVI_std', 'NDWI_mean', 'NDWI_median', 'NDWI_std'])

for meadowIdx in range(meadows.shape[0]):
    # extract a single meadow and it's geometry bounds;
    feature = meadows.loc[meadowIdx, :]
    if feature.geometry.geom_type == 'Polygon':
        shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords))
    elif feature.geometry.geom_type == 'MultiPolygon':
        shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms))
    # convert landsat image collection over each meadow to list for iteration
    target_date = feature.DtFire.strftime("%Y-%m-%d")
    # extract data up to 30 days before fire date (can change date)
    meadow_values = getBandValues(shapefile_bbox, target_date, 30)
    
    if not meadow_values:     # drop rows that returned null band values
        print("Row index", meadowIdx, "dropped!")
        meadow_data.drop(meadowIdx, inplace=True)
        continue
    
    # extract the meadow values and append to dataframe
    band_values, nppVal, time_diff, image_date, image_time = meadow_values 
    meadow_data.iloc[meadowIdx, :] = [feature.UniqueID, feature.FireID, target_date, image_date, image_time, 
                                      time_diff, nppVal] + band_values
    # print progress for every 20 meadows processed
    if meadowIdx%20 == 0: print(meadowIdx, end=', ')

# check how many meadows had data successfully extracted
len([x for x in meadow_data['BSI_mean']])
# reset index for dropped meadow
meadow_data.reset_index(drop=True, inplace=True)
meadow_data.head()
# write updated dataframe to new csv file
meadow_data.to_csv("files/fire_meadows_data.csv", index=False)
    