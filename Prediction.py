# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

@author: jonyegbula
"""

import os
import ee
import numpy as np
import pickle
import calendar
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from geocube.api.core import make_geocube
from joblib import Parallel, delayed
from shapely.geometry import Polygon
import geemap
import multiprocessing
import contextlib
import rioxarray as xr

mydir = "Code"
os.chdir(mydir)
pd.set_option("display.precision", 16)

# Authenticate and initialize python access to Google Earth Engine
# ee.Authenticate()    # only use if you've never run this on your current computer before or loss GEE access
ee.Initialize()

# read in shapefile, landsat and flow accumulation data
shapefile = gpd.read_file("files/AllPossibleMeadows_2024-07-10.shp")
#verify CRS or convert to WGS '84
shapefile.crs
epsg_crs = "EPSG:4326"
shapefile = shapefile.to_crs(epsg_crs)
utm_zone11 = gpd.read_file("files/CA_UTM11.shp")
utm_zone10 = gpd.read_file("files/CA_UTM10.shp")
# zone11_meadows = gpd.overlay(shapefile, utm_zone11, how="intersection") # Repeat all for zone 10 (make changes)
zone10_meadows = gpd.overlay(shapefile, utm_zone10, how="intersection")
shapefile = None

ncores = multiprocessing.cpu_count()
year = 2019
start_year, end_year = str(year-1)+"-10-01", str(year)+"-10-01"
date_range = pd.date_range(start=start_year, end=end_year, freq='D')[:-1]
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']).filterDate(start_year, end_year)
landsat7_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']).filterDate(start_year, end_year)
landsat5_collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']).filterDate(start_year, end_year)
landsat9_collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']).filterDate(start_year, end_year)

flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate('2018-10-01', '2019-10-01').select(['tmmn', 'tmmx', 'pr'])
dem = ee.Image('USGS/3DEP/10m').select('elevation')
slope = ee.Terrain.slope(dem)
daymet = ee.ImageCollection("NASA/ORNL/DAYMET_V4").select('swe')


def maskAndRename(image):
    # rename bands and mask out cloud based on bits in QA_pixel
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


def geotiffToCsv(input_raster, bandnames, crs):
    # creates a dataframe of unique columns (hence combines repeating band names)
    geotiff = xr.open_rasterio(input_raster)
    df = geotiff.to_dataframe(name='value').reset_index()
    df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
    geotiff.close()
    nBands = df.shape[1] - 2
    
    out_csv = pd.DataFrame()
    ncol = len(set(bandnames))
    n = ((nBands - ncol)//11 + 1)
    
    # There are 11 (repeating) landsat, gridmet and constant value bands
    for col in range(1, 12):
        values = []
        for band in [col] + list(range(ncol+col, nBands+1, 11)):
            values = values + list(df[band])
        out_csv[bandnames[col-1]] = values
    
    # repeat the other columns throughout length of dataframe
    for col in range(12, 16):
        out_csv[bandnames[col-1]] = list(df[col])*n
    out_csv['x'] = list(df['x'])*n
    out_csv['y'] = list(df['y'])*n
    
    return out_csv


def interpolate_group(df):
    group = df.reindex(date_range).interpolate(method='linear').ffill().bfill()
    return group
    
    arr1 = np.array([calendar.timegm(d.timetuple()) for d in df['Date']])
    arr2 = np.array(df['CO2.umol.m2.s'])
    interpolated_values = np.interp([calendar.timegm(d.timetuple()) for d in date_range], arr1, arr2)
    return pd.DataFrame({'X': df['X'].iloc[0], 'Y': df['Y'].iloc[0],
                         'Date': date_range, 'CO2.umol.m2.s': interpolated_values})


#load ML GBM models
f = open('files/models.pckl', 'rb')
ghg_model, agb_model, bgb_model = pickle.load(f)
f.close()

landsat_collection = landsat9_collection.merge(landsat8_collection).merge(landsat7_collection).merge(landsat5_collection).map(maskAndRename)
# zone11_meadows['Buffer'] = zone11_meadows.to_crs(32611).geometry.buffer(-30)
# zone11_meadows = zone11_meadows[~zone11_meadows.Buffer.is_empty]
# zone11_meadows.reset_index(drop=True, inplace=True)
zone10_meadows['Buffer'] = zone10_meadows.to_crs(32611).geometry.buffer(-30)
zone10_meadows = zone10_meadows[~zone10_meadows.Buffer.is_empty]
zone10_meadows.reset_index(drop=True, inplace=True)
allIds = zone10_meadows.index   # [54, 96, 1299, 773, 3482, 1024, 1143, 1041, 3465]


def processMeadow(meadowId):
    start = datetime.now()
    # extract a single meadow and it's geometry bounds; buffer inwards to remove trees by edges by designated amount
    feature = zone10_meadows.loc[meadowId, :]
    
    if feature.geometry.geom_type == 'Polygon':
        shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords)).buffer(-30)
    elif feature.geometry.geom_type == 'MultiPolygon':
        shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms)).buffer(-30)
    
    # convert landsat image collection over each meadow to list for iteration
    landsat_images = landsat_collection.filterBounds(shapefile_bbox)
    image_list = landsat_images.toList(landsat_images.size())
    image_result = ee.Dictionary({'crs': landsat_images.aggregate_array('UTM_ZONE'), 'sensor': landsat_images.aggregate_array('SPACECRAFT_ID'), 'image_dates': landsat_images.aggregate_array('system:time_start')}).getInfo()
    dates = [date/1000 for date in image_result['image_dates']]
    mycrs = 'EPSG:326' + str(image_result['crs'][0])
    sensors = [int(sensor[-1]) for sensor in image_result['sensor']]
    noImages = len(dates)
    
    # clip flow, elevation and slope to meadow's bounds
    flow_band = flow_acc.clip(shapefile_bbox)
    dem_bands = dem.clip(shapefile_bbox)
    slopeDem = slope.clip(shapefile_bbox)
    daymetv4 = daymet.filterBounds(shapefile_bbox).filterDate(year + '-04-01', year + '-04-02').first()
    
    # dataframe to store results for each meadow
    cols = ['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Minimum_temperature', 'Maximum_temperature', 'Mean_Precipitation',
            'Date', 'Sensor', 'Flow', 'Elevation', 'Slope', 'SWE', 'X', 'Y', 'NDVI', 'NDWI', 'EVI', 'SAVI', 'BSI', 'NDPI', 'NDSI']
    all_data = pd.DataFrame(columns=cols)
    df = pd.DataFrame()
    combined_image = None
    var_col = []
    bandnames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'tmmn', 'tmmx', 'Date', 'Sensor', 'b1', 'elevation', 'slope']
    subregions = [shapefile_bbox]
    
    # iterate through each landsat image
    for idx in range(noImages):
        landsat_image = ee.Image(image_list.get(idx)).toFloat()
        # use each date in which landsat image exists to extract bands of gridmet
        start_date = timedelta(seconds = dates[idx]) + datetime(1970, 1, 1)
        date = datetime.strftime(start_date, '%Y-%m-%d')
        
        gridmet_filtered = gridmet.filterDate(date, (start_date + timedelta(days=1)).strftime('%Y-%m-%d')).first().clip(shapefile_bbox)
        gridmet_30m = gridmet_filtered.resample('bilinear')
        date_band = ee.Image.cat([ee.Image.constant(dates[idx]).rename('Date'), ee.Image.constant(sensors[idx]).rename('Sensor')])
        
        # align other satellite data with landsat and make resolution uniform (30m)
        if idx == 0:     # only extract once for the same meadow
            daymet_swe = daymetv4.resample('bilinear').toFloat()
            flow_30m = flow_band.resample('bilinear').toFloat()
            dem_30m = dem_bands.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear')
            slope_30m = slopeDem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear')
            combined_image = landsat_image.addBands([gridmet_30m, date_band, flow_30m, dem_30m, slope_30m, daymet_swe])
        else:
            combined_image = combined_image.addBands([landsat_image, gridmet_30m, date_band])
            bandnames = bandnames.copy() + bandnames[:11]     # 11 total of: landsat, gridmet and constant bands
        print(idx, end=' ')
        
    if feature.Area_km2 > 15:     # split bounds of large meadows into smaller regions to stay within limit of image downloads
        xmin, ymin, xmax, ymax = feature.geometry.bounds
        num_subregions = round(np.sqrt(feature.Area_km2/5))
        subregion_width = (xmax - xmin) / num_subregions
        subregion_height = (ymax - ymin) / num_subregions
        subregions = []
        for i in range(num_subregions):
            for j in range(num_subregions):
                subarea = Polygon([(xmin + i*subregion_width, ymin + j*subregion_height),
                                   (xmin + (i+1)*subregion_width, ymin + j*subregion_height),
                                   (xmin + (i+1)*subregion_width, ymin + (j+1)*subregion_height),
                                   (xmin + i*subregion_width, ymin + (j+1)*subregion_height)])
                if subarea.intersects(feature.geometry):
                    subregion = ee.Geometry.Rectangle(list(subarea.bounds))
                    subregions.append(subregion.intersection(shapefile_bbox))    
    # some subregions of small meadows are still large
    else:
        image_name = f'files/meadow_{meadowId}_0.tif'
        subregion = shapefile_bbox
        with contextlib.redirect_stdout(None):
            geemap.ee_export_image(combined_image.clip(subregion), filename=image_name, scale=30, region=subregion, crs=mycrs)
        
        if os.path.exists(image_name):
            df = geotiffToCsv(image_name, bandnames, mycrs)
        else:
            xmin, ymin, xmax, ymax = feature.geometry.bounds
            num_subregions = 2
            subregion_width = (xmax - xmin) / num_subregions
            subregion_height = (ymax - ymin) / num_subregions
            subregions = []
            for i in range(num_subregions):
                for j in range(num_subregions):
                    subarea = Polygon([(xmin + i*subregion_width, ymin + j*subregion_height),
                                       (xmin + (i+1)*subregion_width, ymin + j*subregion_height),
                                       (xmin + (i+1)*subregion_width, ymin + (j+1)*subregion_height),
                                       (xmin + i*subregion_width, ymin + (j+1)*subregion_height)])
                    if subarea.intersects(feature.geometry):
                        subregion = ee.Geometry.Rectangle(list(subarea.bounds))
                        subregions.append(subregion.intersection(shapefile_bbox))
        
    for i, subregion in enumerate(subregions):
        if df.empty:
            temp_image = combined_image.clip(subregion)
            image_name = f'files/meadow_{meadowId}_{i}.tif'
        
            # suppress output of downloaded images (image limit = 48 MB)
            with contextlib.redirect_stdout(None):
                geemap.ee_export_image(temp_image, filename=image_name, scale=30, region=subregion, crs=mycrs)
            try:
                df = geotiffToCsv(image_name, bandnames, mycrs)
            except:
                continue
        nullIds =  list(np.where(df['tmmx'].isnull())[0])
        df.drop(nullIds, inplace = True)
        df.reset_index(drop=True, inplace=True)
        
        # temporary dataframe for each iteration to be appended to the overall dataframe
        df.columns = cols[:-5]
        df['Date'] = pd.to_timedelta(df['Date'], unit='s') + pd.to_datetime('1970-01-01')
        df['NDVI'] = (df['NIR'] - df['Red'])/(df['NIR'] + df['Red'])
        df['NDWI'] = (df['NIR'] - df['SWIR_1'])/(df['NIR'] + df['SWIR_1'])
        df['EVI'] = 2.5*(df['NIR'] - df['Red'])/(df['NIR'] + 6*df['Red'] - 7.5*df['Blue'] + 1)
        df['SAVI'] = 1.5*(df['NIR'] - df['Red'])/(df['NIR'] + df['Red'] + 0.5)
        df['BSI'] = ((df['Red'] + df['SWIR_1']) - (df['NIR'] + df['Blue']))/(df['Red'] + df['SWIR_1'] + df['NIR'] + df['Blue'])
        df['NDPI'] = (df['NIR'] - (0.56*df['Red'] + 0.44*df['SWIR_2']))/(df['NIR'] + 0.56*df['Red'] + 0.44*df['SWIR_2'])
        df['NDSI'] = (df['Green'] - df['SWIR_1'])/(df['Green'] + df['SWIR_1'])
        df.dropna(inplace=True)
        NA_Ids = df.isin([np.inf, -np.inf]).any(axis=1)
        df = df[~NA_Ids]
        all_data = pd.concat([all_data, df])
        df = pd.DataFrame()
    print("\nBand data extraction done!")    
    all_data.head()
    
    if not all_data.empty:
        all_data.drop_duplicates(inplace=True)
        all_data.reset_index(drop=True, inplace=True)
        # select relevant columns and fix order of columns for ML models
        var_col = [c for c in all_data.columns if c not in ['Date', 'Sensor', 'X', 'Y', 'NDSI', 'NDPI', 'SWE', 'Mean_Precipitation']]
        var_col[6:11] = ['Flow', 'Elevation', 'Slope', 'Minimum_temperature', 'Maximum_temperature']
        all_data['CO2.umol.m2.s'] = ghg_model.predict(all_data.loc[:, var_col])
        var_col = [c for c in all_data.columns if c not in ['Date', 'Sensor', 'X', 'Y', 'NDSI', 'Minimum_temperature', 'Maximum_temperature']]
        
        
        # fix interpolation and calculate mean precip, snow and water days, then integrals
        all_data = all_data.groupby(['X', 'Y']).apply(interpolate_group).reset_index(drop=True)
        
        # Predict AGB/BGB using max EVI per pixel (reshaping of values is needed for a single row's prediction)
        all_data.loc[all_data['NDSI'] > 0.2, 'EVI'] = 0
        max_NDVI_Ids = all_data[(4 <= all_data['Date'].dt.month <= 9) & (all_data['NDSI'] <= 0)].groupby(['X', 'Y'])['EVI'].idxmax()
        agb_bgb = pd.DataFrame()
        agb_bgb['HerbBio.g.m2'] = agb_model.predict(all_data.loc[max_NDVI_Ids, var_col])
        agb_bgb['Roots.kg.m2'] = bgb_model.predict(all_data.loc[max_NDVI_Ids, var_col])
        
        # Turn negative AGB/BGB to zero and interpolate GHG    
        agb_bgb.loc[agb_bgb['HerbBio.g.m2'] < 0, 'HerbBio.g.m2'] = 0
        agb_bgb.loc[agb_bgb['Roots.kg.m2'] < 0, 'Roots.kg.m2'] = 0
        all_data = all_data.groupby(['X', 'Y']).apply(interpolate_group).reset_index(drop=True)
        
        # Display summaries for AGB/BGB predictions, then winter/summer summaries for GHG
        agb_bgb.describe()
        GHG = all_data[(all_data.Date.dt.month >= 10) | (all_data.Date.dt.month <= 3)]
        GHG.iloc[:, [-1]].describe()
        GHG = all_data[(all_data.Date.dt.month < 10) & (all_data.Date.dt.month > 3)]
        GHG.iloc[:, [-1]].describe()
        print("Predictions and interpolations done!")
        
        uniquePts = all_data.groupby(['X', 'Y'])
        utm_lons, utm_lats = uniquePts['X'].first().values, uniquePts['Y'].first().values
        res = 30
        # make geodataframe of predictions and projected coordinates as crs; convert to raster
        out_rasters = {1: ['_AGB.tif', 'HerbBio.g.m2'], 2: ['_BGB.tif', 'Roots.kg.m2'], 3: ['_GHG.tif', 'CO2.umol.m2.s']}
        for i in range(1,4):
            out_raster = "files/Image_meadow_" + str(round(meadowId)) + out_rasters[i][0]
            if i == 3:
                pixel_values = pd.Series(uniquePts[out_rasters[i][1]].sum().values, name=out_rasters[i][1])
            else:
                pixel_values = agb_bgb[out_rasters[i][1]]
            gdf = gpd.GeoDataFrame(pixel_values, geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=mycrs.split(":")[1])
            gdf.plot(column=out_rasters[i][1], cmap='viridis', legend=True)
            out_grd = make_geocube(vector_data=gdf, measurements=gdf.columns.tolist()[:-1], resolution=(-res, res))
            out_grd.rio.to_raster(out_raster)
    print(datetime.now() - start)


# processMeadow(96) # 3465 (largest), 54 (smallest)
Parallel(n_jobs=ncores-2)(delayed(processMeadow)(meadowId) for meadowId in allIds)