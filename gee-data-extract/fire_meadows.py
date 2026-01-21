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


# rename bands and mask out cloud based on bits in QA_pixel; then scale values
def maskAndRename(image):
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


# calculate and add indices from landsat band values, and return only indices
def calculateIndices(image):
    ndmi = image.normalizedDifference(['NIR', 'SWIR_1']).rename('NDMI')
    ndwi = image.normalizedDifference(['Green', 'NIR']).rename('NDWI')
    evi = image.expression("2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'BLUE': image.select('Blue')}).rename('EVI')
    bsi = image.expression("((RED + SWIR_1) - (NIR + BLUE)) / (RED + SWIR_1 + NIR + BLUE)", {'RED': image.select('Red'), 'SWIR_1': image.select('SWIR_1'), 'NIR': image.select('NIR'), 'BLUE': image.select('Blue')}).rename('BSI')
    image = image.addBands([ndmi, evi, bsi, ndwi])
    return image.select(["NDMI", "EVI", "BSI", "NDWI"])


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
        # filter the images by dates on or bufferDays before fire date
        temporal_filtered = spatial_filtered.filterDate(ee.Date(target_date).advance(-bufferDays, 'day'), target_date)
        # Sort the images by closest date to fire date and create an image list to loop through if there are valid images
        sorted_collection = temporal_filtered.map(calculate_time_difference).sort('time_difference')
        image_list = sorted_collection.toList(sorted_collection.size())
        noImages = image_list.size().getInfo()
        nImage, band_values = 0, {'NDMI_mean': None}
        reducerFunc = (ee.Reducer.mean().combine(ee.Reducer.median(), sharedInputs=True).combine(ee.Reducer.stdDev(), sharedInputs=True))
        
        # repeatedly check for cloud free pixels (non-null value) in landsat 9, or checks in landsat 8 and then 7
        while band_values['NDMI_mean'] == None and nImage < noImages:
            nearest_image = calculateIndices(ee.Image(image_list.get(nImage)))
            nImage += 1
            properties = nearest_image.getInfo()['properties']
            band_values = nearest_image.reduceRegion(reducerFunc, shapefile_bbox, 30).getInfo()
        
        # number and fraction of pixels for NDWI > 0 and NDWI > 0.3
        water_coverage = nearest_image.select('NDWI').addBands(nearest_image.select('NDWI').gt(0.3).rename(
            'NDWI_03')).addBands(nearest_image.select('NDWI').gt(0.0).rename('NDWI_00')).reduceRegion(
            reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.count(), sharedInputs=True),
                                      geometry=shapefile_bbox, scale=30, maxPixels=1e13).getInfo()
        return [list(band_values.values())[:-3], water_coverage, properties['time_difference'], properties['DATE_ACQUIRED'], properties['SCENE_CENTER_TIME']]
    except:     # if there was error in retrieving values due to no data available within dates
        return []

# read meadow data and extract entire boundary for filtering GEE images
epsg_crs = "EPSG:4326"
meadows = gpd.read_file("files/meadows_dateBeforeFire_fires2012to2023_20251210.shp").to_crs(epsg_crs)
minx, miny, maxx, maxy = meadows.total_bounds
merged_zones = gpd.GeoDataFrame([1], geometry=[box(minx, miny, maxx, maxy)], crs=epsg_crs)
meadow_zone = ee.Geometry.Polygon(list(merged_zones.geometry[0].exterior.coords)).buffer(100)
# sort the meadows by fire date to read pixel NPP data one year at a time
meadows.sort_values(by="DtFire", inplace=True)
meadows.reset_index(drop=True, inplace=True)

# load all landsat images (order by most recent landsat) and npp values
landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat = landsat9.merge(landsat8).merge(landsat7).merge(landsat5).filterBounds(meadow_zone).map(maskAndRename)
landsat_npp = ee.ImageCollection("UMT/NTSG/v2/LANDSAT/NPP").select('annualNPP')

# create empty dataframe, one for each meadow
meadow_data = pd.DataFrame(index=np.arange(meadows.shape[0]), columns=['UniqueID', 'FireID', 'Fire_Date', 'Image_Date',
            'Image_Time', 'Days_Difference', 'NDWI_0_total', 'NDWI_0_fraction', 'NDWI_03_total',
            'NDWI_03_fraction', 'BSI_mean', 'BSI_median', 'BSI_std', 'EVI_mean','EVI_median', 'EVI_std', 'NDMI_mean',
            'NDMI_median', 'NDMI_std', 'ANPP_mean', 'ANPP_std'])

# define the earliest year to read pixel level NPP data from and extract NPP data for all meadows
my_year = str(meadows.DtFire[0].year)
meadows_geom = gpd.GeoDataFrame(geometry=meadows.geometry, crs=meadows.crs)
data = pd.read_csv(f"files/{my_year}_Meadows.csv")
# spatial join between each pixel of csv and meadow's geometry
pixels_gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs="EPSG:4326")    
joined = gpd.sjoin(pixels_gdf, meadows_geom, how='inner', predicate='within')
stats = joined.groupby('index_right')['ANPP'].agg(['mean', 'std'])

for meadowIdx in range(meadows.shape[0]):
    # extract a single meadow and it's geometry bounds;
    feature = meadows.loc[meadowIdx, :]
    if feature.geometry.geom_type == 'Polygon':
        shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords))
    elif feature.geometry.geom_type == 'MultiPolygon':
        shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms))
    # convert landsat image collection over each meadow to list for iteration
    try:
        target_date = feature.DtFire.strftime("%Y-%m-%d")
        year, month, day = target_date.split("-")
    except:     # skip null dates
        continue
    
    # read a new pixel level csv file when the next year is encountered (saves computational cost)
    if year != my_year:
        my_year = year
        data = pd.read_csv(f"files/{my_year}_Meadows.csv")
        pixels_gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs="EPSG:4326")    
        joined = gpd.sjoin(pixels_gdf, meadows_geom, how='inner', predicate='within')
        stats = joined.groupby('index_right')['ANPP'].agg(['mean', 'std'])
        
    # extract data up to 30 days before fire date (can change date)
    meadow_values = getBandValues(shapefile_bbox, target_date, 30)
    if not meadow_values:     # skip rows that returned null band values
        continue
    
    try:    # check if valid NPPs are available for meadow
        NPP = stats.loc[meadowIdx, :]
        NPP_vals = [NPP['mean'],  NPP['std']]
    except:
        NPP_vals = [None, None]
    
    # extract the meadow values and append to dataframe
    band_values, NDWI_pixels, time_diff, image_date, image_time = meadow_values
    NDWI_pixels = list(NDWI_pixels.values())[:4]
    if NDWI_pixels[1]: NDWI_pixels[0], NDWI_pixels[2] = round(NDWI_pixels[1] * NDWI_pixels[0]), round(NDWI_pixels[2] * NDWI_pixels[3])
    meadow_data.iloc[meadowIdx, :] = [feature.UniqueID, feature.FireID, target_date, image_date, image_time, 
                                      time_diff] + NDWI_pixels + band_values + NPP_vals
    # print progress for every 20 meadows processed
    if meadowIdx%20 == 0: print(meadowIdx, end=', ')

# check how many meadows had data successfully extracted
len([x for x in meadow_data['BSI_mean'] if x])
meadow_data.sort_values(by="UniqueID", inplace=True)
meadow_data.head()
# write updated dataframe to new csv file
meadow_data.to_csv("csv/fire_meadows_data.csv", index=False)
