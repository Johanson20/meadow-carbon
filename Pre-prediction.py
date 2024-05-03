# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

@author: jonyegbula
"""

import os
import ee
import pickle
import warnings
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from geocube.api.core import make_geocube
from pyproj import Proj, transform

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
meadowId = 9313    # 9313 (crosses image boundary), 17902 (largest), 16658 (smallest)
feature = shapefile.loc[meadowId, ]
if feature.geometry.geom_type == 'Polygon':
    shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords)).buffer(-5)
elif feature.geometry.geom_type == 'MultiPolygon':
    shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms)).buffer(-5)

# convert landsat image collection over each meadow to list for iteration
landsat_images = landsat8_collection.filterBounds(shapefile_bbox).map(maskImage)
image_list = landsat_images.toList(landsat_images.size())
noImages = image_list.size().getInfo()

# clip flow, elevation and slope to meadow's bounds
flow_band = flow_acc.clip(shapefile_bbox)
dem_bands = dem.clip(shapefile_bbox)
slopeDem = ee.Terrain.slope(dem).clip(shapefile_bbox)

# dataframe to store results for each meadow
cols = ['Date', 'Pixel', 'Longitude', 'Latitude', 'Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Flow', 'Elevation', 'Slope', 
        'Minimum_temperature', 'Maximum_temperature', 'NDVI', 'NDWI', 'EVI', 'SAVI', 'BSI', 'CO2.umol.m2.s', 'HerbBio.g.m2', 'Roots.kg.m2']
all_data = pd.DataFrame(columns=cols)
flow_values, latlon, lat = [], None, []

# iterate through each landsat image
for idx in range(noImages):
    # extract pixel coordinates and band values from landsat
    landsat_image = ee.Image(image_list.get(idx))
    
    if not flow_values:     # only extract once for the same meadow
        flow_30m = flow_band.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
        dem_30m = dem_bands.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
        slope_30m = slopeDem.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
        flow_values = flow_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['b1']
        elev_values = dem_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['elevation']
        slope_values = slope_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['slope']
        n = len(flow_values)
    
    if not latlon or len(lat) != n:      # only extract pixel coordinates once as value is constant for the same meadow
        latlon = landsat_image.sample(region=shapefile_bbox, scale=30, geometries=True).getInfo()['features']
        if not lat or len(lat) != n:      # if no coordinate found, skip iteration (probably cloud/snow cover)
            lat = [feat['geometry']['coordinates'][1] for feat in latlon]
            lon = [feat['geometry']['coordinates'][0] for feat in latlon]
    band_values = landsat_image.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()
    
    if n == 0 or n != len(band_values['SR_B2']):    # usually less landsat values than flow value is due to cloud/snow mask
        continue
    print(idx, end=' ')
    
    # use each date in which landsat image exists to extract bands of gridmet, flow and DEM
    date = landsat_image.getInfo()['properties']['DATE_ACQUIRED']
    start_date = datetime.strptime(date, '%Y-%m-%d')
    gridmet_filtered = gridmet.filterDate(date, (start_date + timedelta(days=1)).strftime('%Y-%m-%d')).first()
    
    # align other satellite data with landsat and make resolution uniform (30m)
    gridmet_30m = gridmet_filtered.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30).clip(shapefile_bbox)
    min_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['tmmn']
    max_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['tmmx']
    
    # temporary dataframe for each iteration to be appended to the overall dataframe
    df = pd.DataFrame(columns=cols[:-3])
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
    df['CO2.umol.m2.s'] = ghg_model.predict(df.iloc[:, 4:])
    all_data = pd.concat([all_data, df])

all_data.head()
# append coordinates and convert column data types for further processing
if not all_data.empty:
    n = all_data.shape[0]//df.shape[0]
    all_data['Longitude'] = lon*n
    all_data['Latitude'] = lat*n
    all_data['Date'] = pd.to_datetime(all_data['Date'], format="%Y-%m-%d")
    all_data[cols[1:]] = all_data[cols[1:]].apply(pd.to_numeric)
    all_data.reset_index(drop=True, inplace=True)
    
    # get indices of max NIR to predict AGB/BGB
    maxIds = all_data.groupby('Pixel')['NIR'].idxmax()
    all_data['HerbBio.g.m2'] = list(agb_model.predict(all_data.iloc[maxIds, 4:-3]))*n
    all_data['Roots.kg.m2'] = list(bgb_model.predict(all_data.iloc[maxIds, 4:-3]))*n

all_data.head()
# predict on dataframe
if not all_data.empty:
    # convert to projected coordinates so that resolution would be meaningful
    lats = all_data['Latitude'].values
    lons = all_data['Longitude'].values
    utm_lons, utm_lats = transform(Proj('EPSG:4326'), Proj('EPSG:32610'), lats, lons)
    res = 30
    out_raster = "Image_" + str(round(feature.ID)) + "_" + str(round(feature.Area_m2)) + ".tif"
    
    # make geodataframe of relevant columns and crs of projected coordinates
    gdf = gpd.GeoDataFrame(all_data.iloc[:, 2:], geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=32610)
    gdf.plot()
    # make a grid (to be converted to a geotiff) where each column is a band (exclude geometry column)
    out_grd = make_geocube(vector_data=gdf, measurements=gdf.columns.tolist()[:-1], resolution=(-res, res))
    out_grd.rio.to_raster(out_raster)
