# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 15:09:32 2025

@author: jonyegbula
"""

import os
import time
import numpy as np
import pickle
import warnings
import pandas as pd
import geopandas as gpd
import multiprocessing
import contextlib
import rasterio
import rioxarray as xr
import ee
import geemap
from matplotlib.colors import ListedColormap
from datetime import datetime
from dateutil.relativedelta import relativedelta
from geocube.api.core import make_geocube
from shapely.geometry import box, Polygon
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)
warnings.filterwarnings("ignore")
folder_id = "1RpZRfWUz6b7UpZfRByWSXuu0k78BAxzz"  
ee.Initialize()


def maskAndRename(image):
    # rename bands and mask out cloud based on bits in QA_pixel; then scale values
    image = image.rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'QA'])
    qa = image.select('QA')
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = dilated_cloud.And(cirrus).And(cloud).And(cloudShadow)
    image = image.updateMask(cloud_mask).select(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2'])
    scaled_bands = image.multiply(2.75e-05).add(-0.2)
    return image.addBands(scaled_bands, overwrite=True)


# Calculates absolute time difference (in days) from a target date, in which the images are acquired
def calculate_time_difference(image):
    time_difference = ee.Number(image.date().difference(target_date, 'day')).abs()
    return image.set('time_difference', time_difference)


def calculateIndices(image):
    # calculate and add indices from landsat band values
    ndwi = image.normalizedDifference(['NIR', 'SWIR_1']).rename('NDWI')
    evi = image.expression("2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'BLUE': image.select('Blue')}).rename('EVI')
    bsi = image.expression("((RED + SWIR_1) - (NIR + BLUE)) / (RED + SWIR_1 + NIR + BLUE)", {'RED': image.select('Red'), 'SWIR_1': image.select('SWIR_1'), 'NIR': image.select('NIR'), 'BLUE': image.select('Blue')}).rename('BSI')
    return image.addBands([ndwi, evi, bsi])


# Function to extract cloud free band values per pixel from landsat 8 or landsat 7
def getBandValues(boundary, target_date, bufferDays = 30):
    # filter landsat images by location
    spatial_filtered = landsat.filterBounds(boundary)
    # filter the streamlined images by dates +/- a certain number of days
    temporal_filtered = spatial_filtered.filterDate(ee.Date(target_date).advance(-bufferDays, 'day'), target_date)
    # Map the ImageCollection over time difference and sort by that property
    sorted_collection = temporal_filtered.map(calculate_time_difference).sort('time_difference')
    image_list = sorted_collection.toList(sorted_collection.size())
    noImages = image_list.size().getInfo()
    nImage, band_values = 0, {'Red': None}
    
    # repeatedly check for cloud free pixels (non-null value) in landsat 8, or checks in landsat 7
    while band_values['Red'] == None and nImage < noImages:
        nearest_image = ee.Image(image_list.get(nImage))
        nImage += 1
        properties = nearest_image.getInfo()['properties']
        band_values = nearest_image.reduceRegion(ee.Reducer.mean(), boundary, 30).getInfo()
    
    return [list(band_values.values()), properties['time_difference'], properties['DATE_ACQUIRED'], properties['SCENE_CENTER_TIME']]


epsg_crs = "EPSG:4326"
meadows = gpd.read_file("files/meadows_dateBeforeFire_fires2012to2023_20251210.shp").to_crs(epsg_crs)
minx, miny, maxx, maxy = meadows.total_bounds
merged_zones = gpd.GeoDataFrame([1], geometry=[box(minx, miny, maxx, maxy)], crs=epsg_crs)
sierra_zone = ee.Geometry.Polygon(list(merged_zones.geometry[0].exterior.coords)).buffer(100)

# load all relevant GEE images/collections for both UTM Zones
landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat = landsat9.merge(landsat8).merge(landsat7).merge(landsat5).filterBounds(sierra_zone).map(maskAndRename)


for meadowIdx in range(len(meadows.shape[0])):
    # extract a single meadow and it's geometry bounds; buffer inwards to remove edge effects
    feature = meadows.loc[meadowIdx, :]
    if feature.geometry.geom_type == 'Polygon':
        if feature.AreaKm2 > 0.5:
            feature.geometry = feature.geometry.simplify(0.00001)
        shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords))
    elif feature.geometry.geom_type == 'MultiPolygon':
        shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms))
    
    # convert landsat image collection over each meadow to list for iteration
    landsat_images = getBandValues(shapefile_bbox, feature.DtFire.strftime("%Y-%m-%d"))
    image_list = landsat_images.toList(landsat_images.size())
    try:
        image_result = ee.Dictionary({'image_dates': landsat_images.aggregate_array('system:time_start')}).getInfo()
    except:
        time.sleep(1.1)

