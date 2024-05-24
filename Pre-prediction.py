# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

@author: jonyegbula
"""

import os
import ee
import numpy as np
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
shapefile = gpd.read_file("files/AllPossibleMeadows_2024-02-12.shp")
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate('2018-10-01', '2019-10-01')
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate('2018-10-01', '2019-10-01').select(['pr', 'tmmn', 'tmmx'])
dem = ee.Image('USGS/3DEP/10m').select('elevation')
slope = ee.Terrain.slope(dem)

def maskImage(image):
    qa = image.select('QA_PIXEL')
    # mask out cloud based on bits in QA_pixel
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = dilated_cloud.And(cirrus).And(cloud).And(cloudShadow)
    return image.updateMask(cloud_mask)

#load ML GBM models
f = open('files/models.pckl', 'rb')
ghg_model, agb_model, bgb_model = pickle.load(f)
f.close()

#verify CRS or convert to WGS '84
shapefile.crs

# extract a single meadow and it's geometry bounds; buffer inwards by designated amount
meadowId = 5    # 9313 (crosses image boundary), 17902 (largest), 16658 (smallest)
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
slopeDem = slope.clip(shapefile_bbox)

# dataframe to store results for each meadow
cols = ['Date', 'Pixel', 'Longitude', 'Latitude', 'Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Flow', 'Elevation', 'Slope', 
        'Precipitation', 'Minimum_temperature', 'Maximum_temperature', 'NDVI', 'NDWI', 'EVI', 'SAVI', 'BSI', 'CO2.umol.m2.s',
        'HerbBio.g.m2', 'Roots.kg.m2']
all_data = pd.DataFrame(columns=cols)
flow_values, elev_values, new_bbox = [], [], set()
latlon, lat = None, []

# iterate through each landsat image
for idx in range(noImages):
    # extract pixel coordinates and band values from landsat
    landsat_image = ee.Image(image_list.get(idx))
    # use each date in which landsat image exists to extract bands of gridmet, flow and DEM
    date = landsat_image.getInfo()['properties']['DATE_ACQUIRED']
    start_date = datetime.strptime(date, '%Y-%m-%d')
    gridmet_filtered = gridmet.filterDate(date, (start_date + timedelta(days=1)).strftime('%Y-%m-%d')).first().clip(shapefile_bbox)
    
    # align other satellite data with landsat and make resolution uniform (30m)
    gridmet_30m = gridmet_filtered.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
    if not flow_values:     # only extract once for the same meadow
        flow_30m = flow_band.resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
        dem_30m = dem_bands.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
        slope_30m = slopeDem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear').reproject(crs=landsat_image.projection(), scale=30)
        flow_values = flow_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['b1']
        n = len(flow_values)
    
    if n < 5000:    # 5000 is GEE's request limit
        band_values = landsat_image.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()
        precip = gridmet_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['pr']
        min_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['tmmn']
        max_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['tmmx']
        if not latlon or len(lat) != n:      # only extract pixel coordinates once as value is constant for the same meadow
            latlon = landsat_image.sample(region=shapefile_bbox, scale=30, geometries=True).getInfo()['features']
            if not lat or len(lat) != n:      # if no coordinate found, skip iteration (probably cloud/snow cover)
                lat = [feat['geometry']['coordinates'][1] for feat in latlon]
                lon = [feat['geometry']['coordinates'][0] for feat in latlon]        
        if not elev_values:     # only extract once for the same meadow
            elev_values = dem_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['elevation']
            slope_values = slope_30m.reduceRegion(ee.Reducer.toList(), shapefile_bbox, 30).getInfo()['slope']
    else:
        if not new_bbox:    # split bounds of shapefile so that subregions have less than 5000 pixels
            coords = shapefile_bbox.bounds().coordinates().getInfo()[0]
            xmin, ymin = coords[0]
            xmax, ymax = coords[2]
            num_subregions = round(np.sqrt(n/1250))   # half the dimensions of 5000 pixels for safety 
    
            subregion_width = (xmax - xmin) / num_subregions
            subregion_height = (ymax - ymin) / num_subregions
            subregions = []
            for i in range(num_subregions):
                for j in range(num_subregions):
                    subregion = ee.Geometry.Rectangle([xmin + i*subregion_width, ymin + j*subregion_height,
                                                       xmin + (i+1)*subregion_width, ymin + (j+1)*subregion_height])
                    subregions.append(subregion.intersection(shapefile_bbox))
    
            # iterate over subregions to extract all pixel coordinates from landsat and flow (often same as gridmet and elevation)
            latlon_flow, latlon_landsat = [], []
            count = 0
            for subregion in subregions:
                samples1 = flow_30m.sample(region=subregion, scale=30, geometries=True).getInfo()['features']
                samples2 = landsat_image.sample(region=subregion, scale=30, geometries=True).getInfo()['features']
                if samples1:
                    latlon_flow.extend([coords['geometry']['coordinates'] for coords in samples1])
                if samples2:
                    latlon_landsat.extend([coords['geometry']['coordinates'] for coords in samples2])
                count += 1
                if count % 10 == 0: print(count, end= ' ')
            print()
            
            # a multipoint from the common coordinates defines the new bounding box for band value extraction
            latlon_flow = set([tuple(x) for x in latlon_flow])
            latlon_landsat = set([tuple(x) for x in latlon_landsat])
            latlon = list(latlon_flow.intersection(latlon_landsat))
            lat = [feat[1] for feat in latlon]
            lon = [feat[0] for feat in latlon]
            new_bbox = ee.Geometry.MultiPoint(latlon)
            
            # If EEException results, split new_bbox into two parts to reduce payload
            try:
                flow_values = flow_30m.reduceRegion(ee.Reducer.toList(), new_bbox, 30).getInfo()['b1']
                elev_values = dem_30m.reduceRegion(ee.Reducer.toList(), new_bbox, 30).getInfo()['elevation']
                slope_values = slope_30m.reduceRegion(ee.Reducer.toList(), new_bbox, 30).getInfo()['slope']        
            except:
                point_bbox = ee.Geometry.MultiPoint(latlon[:int(n/2)])
                flow_values = flow_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['b1']
                elev_values = dem_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['elevation']
                slope_values = slope_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['slope']
                point_bbox = ee.Geometry.MultiPoint(latlon[int(n/2):])
                flow_values.extend(flow_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['b1'])
                elev_values.extend(dem_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['elevation'])
                slope_values.extend(slope_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['slope'])
            n = len(flow_values)
        # If EEException results, split new_bbox into two parts to reduce payload
        try:
            band_values = landsat_image.reduceRegion(ee.Reducer.toList(), new_bbox, 30).getInfo()
            precip = gridmet_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['pr']
            min_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), new_bbox, 30).getInfo()['tmmn']
            max_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), new_bbox, 30).getInfo()['tmmx']
        except:
            point_bbox = ee.Geometry.MultiPoint(latlon[:int(n/2)])
            band_values = landsat_image.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()
            precip = gridmet_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['pr']
            min_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['tmmn']
            max_temp = gridmet_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['tmmx']
            point_bbox = ee.Geometry.MultiPoint(latlon[int(n/2):])
            band_values2 = landsat_image.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()
            precip = gridmet_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['pr']
            min_temp.extend(gridmet_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['tmmn'])
            max_temp.extend(gridmet_30m.reduceRegion(ee.Reducer.toList(), point_bbox, 30).getInfo()['tmmx'])
            
            # combine bands for landsat
            for band in band_values:
                band_values[band].extend(band_values2[band])
            
    if n == 0 or n != len(band_values['SR_B2']):    # usually less landsat values than flow value is due to cloud/snow mask
        continue
    print(idx, end=' ')
    
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
    df['Precipitation'] = precip
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
    out_raster = "files/Image_" + str(round(feature.ID)) + "_" + str(round(feature.Area_m2)) + ".tif"
    
    # make geodataframe of relevant columns and crs of projected coordinates
    gdf = gpd.GeoDataFrame(all_data.iloc[:, 2:], geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=32610)
    gdf.plot()
    # make a grid (to be converted to a geotiff) where each column is a band (exclude geometry column)
    out_grd = make_geocube(vector_data=gdf, measurements=gdf.columns.tolist()[:-1], resolution=(-res, res))
    out_grd.rio.to_raster(out_raster)

shapefile = None
