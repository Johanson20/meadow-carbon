# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

@author: jonyegbula
"""

import os, ee
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
os.chdir("Code")

# Authenticate and initialize python access to Google Earth Engine
# ee.Authenticate()    # only use if you've never run this on your current computer before or loss GEE access
ee.Initialize()

# read in shapefile, landsat and flow accumulation data
shapefile = gpd.read_file("AllPossibleMeadows_2024-02-12.shp")
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate('2018-10-01', '2019-10-01')
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate('2018-10-01', '2019-10-01').select(['tmmn', 'tmmx'])
dem = ee.Image('USGS/SRTMGL1_003').select('elevation')

#verify CRS or convert to WGS '84
shapefile.crs

meadowId = 1
feature = shapefile.loc[meadowId, :]['geometry']
if feature.geom_type == 'Polygon':
    shapefile_bbox = ee.Geometry.Polygon(list(feature.exterior.coords)).buffer(30)
elif feature.geom_type == 'MultiPolygon':
    shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geoms)).buffer(30)

landsat_images = landsat8_collection.filterBounds(shapefile_bbox)

image_list = landsat_images.toList(landsat_images.size())
noImages = image_list.size().getInfo()

flow_band = flow_acc.clip(shapefile_bbox)
dem_bands = dem.clip(shapefile_bbox)
slopeDem = ee.Terrain.slope(dem).clip(shapefile_bbox)

all_data = pd.DataFrame(columns=['Pixel', 'Date', 'Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Flow', 'Elevation', 'Slope',
                           'Minimum_temperature', 'Maximum_temperature', 'NDVI', 'NDWI', 'EVI', 'SAVI', 'BSI'])

for idx in range(noImages):
    landsat_image = ee.Image(image_list.get(idx))
    date = landsat_image.getInfo()['properties']['DATE_ACQUIRED']
    start_date = datetime.strptime(date, '%Y-%m-%d')
    gridmet_filtered = gridmet.filterDate(date, (start_date + timedelta(days=1)).strftime('%Y-%m-%d')).first()
    gridmet_30m = gridmet_filtered.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30).clip(shapefile_bbox)
    
    flow_30m = flow_band.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
    dem_30m = dem_bands.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
    slope_30m = slopeDem.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
    
    band_values = landsat_image.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()
    flow_values = flow_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['b1']
    elev_values = dem_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['elevation']
    slope_values = slope_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['slope']
    min_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['tmmn']
    max_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['tmmx']

    df = pd.DataFrame(columns=['Pixel', 'Date', 'Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Flow', 'Elevation', 'Slope',
                               'Minimum_temperature', 'Maximum_temperature', 'NDVI', 'NDWI', 'EVI', 'SAVI', 'BSI'])
    n = len(flow_values)
    
    df['Date'] = [date]*n
    df['Pixel'] = list(range(1, n+1))
    df['Blue'] = band_values['SR_B2']
    df['Green'] = band_values['SR_B3']
    df['Red'] = band_values['SR_B4']
    df['NIR'] = band_values['SR_B5']
    df['SWIR_1'] = band_values['SR_B6']
    df['SWIR_2'] = band_values['SR_B7']
    df['Flow'] = flow_values
    df['Elevation'] = elev_values
    df['Slope'] = slope_values
    df['Minimum_temperature'] = min_temp
    df['Maximum_temperature'] = max_temp
    df['NDVI'] = (df['NIR'] - df['Red'])/(df['NIR'] + df['Red'])
    df['NDWI'] = (df['Green'] - df['NIR'])/(df['Green'] + df['NIR'])
    df['EVI'] = 2.5*(df['NIR'] - df['Red'])/(df['NIR'] + 6*df['Red'] - 7.5*df['Blue'] + 1)
    df['SAVI'] = 1.5*(df['NIR'] - df['Red'])/(df['NIR'] + df['Red'] + 0.5)
    df['BSI'] = ((df['Red'] + df['SWIR_1']) - (df['NIR'] + df['Red']))/(df['Red'] + df['SWIR_1'] + df['NIR'] + df['Red'])
    
    all_data = pd.concat([all_data, df])

all_data['CO2.umol.m2.s'] = ghg_model.predict(all_data.iloc[:, 2:])
all_data.head()
