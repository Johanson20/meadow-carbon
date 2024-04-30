# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

@author: jonyegbula
"""

import os, ee, pickle
import warnings
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
os.chdir("Code")
warnings.filterwarnings("ignore")

# Authenticate and initialize python access to Google Earth Engine
# ee.Authenticate()    # only use if you've never run this on your current computer before or loss GEE access
ee.Initialize()

# read in shapefile, landsat and flow accumulation data
shapefile = gpd.read_file("AllPossibleMeadows_2024-02-12.shp")
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate('2018-10-01', '2019-10-01')
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate('2018-10-01', '2019-10-01').select(['tmmn', 'tmmx'])
dem = ee.Image('USGS/SRTMGL1_003').select('elevation')

def maskImage(image):
    quality = image.select('QA_PIXEL')
    cloud = quality.bitwiseAnd(1 << 3).eq(0)    # mask out cloudy pixels
    cloudShadow = quality.bitwiseAnd(1 << 4).eq(0)     # mask out cloud shadow
    snow = quality.bitwiseAnd(1 <<5).eq(0)     # mask out snow
    return image.updateMask(cloud).updateMask(cloudShadow).updateMask(snow)

#load ML GBM models
f = open('models.pckl', 'rb')
ghg_model, agb_model, bgb_model = pickle.load(f)
f.close()

#verify CRS or convert to WGS '84
shapefile.crs

# extract a single meadow and it's geometry bounds; buffer inwards by designated amount
meadowId = 5    # 9313 (crosses), 17902 (largest), 16658 (smallest)
feature = shapefile.loc[meadowId, 'geometry']
if feature.geom_type == 'Polygon':
    shapefile_bbox = ee.Geometry.Polygon(list(feature.exterior.coords)).buffer(-5)
elif feature.geom_type == 'MultiPolygon':
    shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geoms)).buffer(-5)

# convert landsat image collection over each meadow to list for iteration
landsat_images = landsat8_collection.filterBounds(shapefile_bbox).map(maskImage)
image_list = landsat_images.toList(landsat_images.size())
noImages = image_list.size().getInfo()

# clip flow, elevation and slope to meadow's bounds
flow_band = flow_acc.clip(shapefile_bbox)
dem_bands = dem.clip(shapefile_bbox)
slopeDem = ee.Terrain.slope(dem).clip(shapefile_bbox)

# dataframe to store results for each meadow
cols = ['Pixel', 'Date', 'Longitude', 'Latitude', 'Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Flow', 'Elevation', 'Slope', 
        'Minimum_temperature', 'Maximum_temperature', 'NDVI', 'NDWI', 'EVI', 'SAVI', 'BSI', 'CO2.umol.m2.s', 'HerbBio.g.m2', 'Roots.kg.m2']
all_data = pd.DataFrame(columns=cols)
flow_values = None

# iterate through each landsat image
for idx in range(noImages):
    # extract pixel coordinates and band values from landsat
    landsat_image = ee.Image(image_list.get(idx))
    latlon = landsat_image.sample(region=shapefile_bbox, scale=30, geometries=True).getInfo()['features']
    if not latlon:
        continue
    lat = [feat['geometry']['coordinates'][1] for feat in latlon]
    lon = [feat['geometry']['coordinates'][0] for feat in latlon]
    band_values = landsat_image.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()
    
    # use each date in which landsat image exists to extract bands of gridmet, flow and DEM
    date = landsat_image.getInfo()['properties']['DATE_ACQUIRED']
    start_date = datetime.strptime(date, '%Y-%m-%d')
    gridmet_filtered = gridmet.filterDate(date, (start_date + timedelta(days=1)).strftime('%Y-%m-%d')).first()
    
    # align other satellite data with landsat and make resolution uniform (30m)
    gridmet_30m = gridmet_filtered.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30).clip(shapefile_bbox)
    min_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['tmmn']
    max_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['tmmx']
    
    if not flow_values:
        flow_30m = flow_band.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
        dem_30m = dem_bands.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
        slope_30m = slopeDem.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
        flow_values = flow_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['b1']
        elev_values = dem_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['elevation']
        slope_values = slope_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['slope']

    # temporary dataframe for each iteration to be appended to the overall dataframe
    df = pd.DataFrame(columns=cols[:-3])
    n = len(flow_values)
    
    if n != len(latlon):
        continue
    
    df['Date'] = [date]*n
    df['Pixel'] = list(range(1, n+1))
    df['Longitude'] = lon
    df['Latitude'] = lat
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
    
    # get row of maximum NIR for predicting AGB and BGB
    maxNIR = df[df['NIR'] == max(df['NIR'])].index
    df['CO2.umol.m2.s'] = ghg_model.predict(df.iloc[:, 4:])
    df['HerbBio.g.m2'] = [agb_model.predict(df.iloc[maxNIR[0], 4:-1].values.reshape(1,-1))[0]]*n
    df['Roots.kg.m2'] = [bgb_model.predict(df.iloc[maxNIR[0], 4:-2].values.reshape(1,-1))[0]]*n
    all_data = pd.concat([all_data, df])

# predict on dataframe
all_data.head()
