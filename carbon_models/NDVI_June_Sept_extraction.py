# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

@author: Johanson C. Onyegbula
"""

import os
import time
import numpy as np
import warnings
import pandas as pd
import geopandas as gpd
import contextlib
import rasterio
import rioxarray as xr
import ee
import geemap
from joblib import Parallel, delayed
from datetime import datetime
from shapely.geometry import box

mydir = "Code"      # adjust directory suitably (one folder up based on paths of other files being created)
os.chdir(mydir)
warnings.filterwarnings("ignore")
folder_id = "1RpZRfWUz6b7UpZfRByWSXuu0k78BAxzz"
ee.Initialize()


def geotiffToDataFrame(image_name, cols):
    ''' creates a single dataframe combining both NDVI columns '''
    with rasterio.Env(CPL_LOG='ERROR'):
        geotiff = xr.open_rasterio(image_name)
    df = geotiff.to_dataframe(name='value').reset_index()
    df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
    geotiff.close()
    out_df = pd.DataFrame()
    for col in range(2):
        out_df[cols[col]] = list(df[col+1])
    out_df['X'] = list(df['x'])
    out_df['Y'] = list(df['y'])
    return out_df


def processGeotiff(df):
    ''' cleans up dataframe: drop null columns and convert infinity to NAs '''
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.groupby(['X', 'Y']).apply(interpolate_pixel_group)
    nan_rows = df[df.isna().any(axis=1)]
    if not nan_rows.empty:
        df = df.apply(lambda row: spatial_interpolate(row, df) if row.isna().any() else row, axis=1)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def interpolate_pixel_group(group):
    ''' interpolate NA rows (landsat 7 scan lines) with nearest different date of same pixel '''
    return group.interpolate(method='linear', axis=0, limit_direction='both')


def spatial_interpolate(row, df):
    ''' interpolate NA rows (landsat 7 scan lines) based on median of other pixels on same date '''
    nearest_values = df[(df['Date'] == row['Date']) & ((df['X'] != row['X']) | (df['Y'] != row['Y']))].dropna()
    if not nearest_values.empty:
        row.fillna(nearest_values.median(), inplace=True)
    else:
        other_values = df[(df['X'] != row['X']) | (df['Y'] != row['Y'])].dropna()
        if not other_values.empty:  # use ~30 day radius of different pixel
            next_close_values = other_values[abs(other_values['Date'] - row['Date']) < 2.6e6]
            if not next_close_values.empty:
                row.fillna(next_close_values.median(), inplace=True)
    return row


def interpolate_group(group):
    ''' returns first instance of pixel/group, otherwise deletes the row '''
    try:
        return group.head(1)
    except:
        return pd.DataFrame()


def maskAndRename(image):
    ''' rename bands and mask out cloud based on bits in QA_pixel; then scale values '''
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


def calculateIndices(image):
    ''' calculate and adds NDVI from landsat band values '''
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    return image.addBands(ndvi)


def loadYearCollection(year):
    ''' load temporal GEE data, generally re-used variables for the relevant year and make folders for each year '''
    global start_year, end_year, landsat_5_year, landsat_collection
    start_year, end_year = str(year-1)+"-10-01", str(year)+"-10-01"
    landsat_collection = landsat.filterDate(start_year, end_year)
    os.makedirs(f"files/{year}", exist_ok=True)
    os.makedirs(f"files/NDVIs/{year}", exist_ok=True)


def downloadImageBands(subregion, imagename, feature, combined_image):
    ''' downloads multi-band geotiffs directly to local storage '''
    mycrs = feature.epsgCode
    # either directly download images of small meadows locally or export large ones to google drive before downloading locally
    image_name = f'{imagename}.tif'
    if not os.path.exists(image_name):     # (image limit = 48 MB)
        with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
            geemap.ee_export_image(combined_image.clip(subregion), filename=image_name, scale=30, crs=mycrs, region=subregion)
            if not os.path.exists(image_name):
                time.sleep(1.1)
                with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
                    geemap.ee_export_image(combined_image.clip(subregion), filename=image_name, scale=30, crs=mycrs, region=subregion)
    return feature.ID


def generateCombinedImage(shapefile_bbox):
    ''' combine June and Sept NDVI bands into one multi-band image within meadow's bounds '''
    landsat_5_year = landsat.filterDate(str(int(year)-6)+"-10-01", str(year-1)+"-10-01").filterBounds(shapefile_bbox).map(calculateIndices)
    landsat_June = landsat_5_year.select('NDVI').filter(ee.Filter.calendarRange(6, 6, 'month')).mean()
    landsat_Sept = landsat_5_year.select('NDVI').filter(ee.Filter.calendarRange(9, 9, 'month')).mean()
    combined_image = landsat_June.addBands(landsat_Sept)
    return combined_image


# resample and reproject when image's pixel size is not 30m for both UTM zones
def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=30)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=30)


# read in shapefile, landsat and flow accumulation data and convert shapefile to WGS '84
epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/AllPossibleMeadows_2025-10-22.shp").to_crs(epsg_crs)
# file handles need to be closed for serialization of parallel processes
shapefile['epsgCode'] = "EPSG:32611"
utm_zone10 = gpd.read_file("files/CA_UTM10.shp").to_crs(epsg_crs)
allIds = list(gpd.overlay(shapefile, utm_zone10, how="intersection").ID)
shapefile.loc[shapefile['ID'].isin(allIds), 'epsgCode'] = "EPSG:32610"
# add a buffer of 100m to Sierra Nevada
minx, miny, maxx, maxy = shapefile.total_bounds
merged_zones = gpd.GeoDataFrame([1], geometry=[box(minx, miny, maxx, maxy)], crs=epsg_crs)
sierra_zone = ee.Geometry.Polygon(list(merged_zones.geometry[0].exterior.coords)).buffer(100)

# load all Landsat images/collections
warnings.filterwarnings("ignore", category=DeprecationWarning)
landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat = landsat9.merge(landsat8).merge(landsat7).merge(landsat5).filterBounds(sierra_zone).map(maskAndRename)

cols = ['NDVI_June', 'NDVI_Sept', 'X', 'Y']
allIds = shapefile.ID


def prepareMeadows(meadowId):
    ''' function to search GEE dataset and extract NDVI bands, combine them and download geotiffs '''
    try:
        # extract a single meadow and it's geometry bounds; buffer inwards to remove edge effects
        feature = shapefile[shapefile.ID == meadowId].iloc[0]
        if feature.geometry.geom_type == 'Polygon':
            if feature.Area_km2 > 0.5:
                feature.geometry = feature.geometry.simplify(0.00001)
            shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords)).buffer(-30)
        elif feature.geometry.geom_type == 'MultiPolygon':
            shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms)).buffer(-30)
        # combine bands and split large meadows
        combined_image = generateCombinedImage(shapefile_bbox)
        imagename = f'files/NDVIs/{year}/meadow_{year}_{meadowId}'
        isValidBand = downloadImageBands(shapefile_bbox, imagename, feature, combined_image)
        return isValidBand
    except:
        return -4


def processMeadow(meadowId):
    ''' function to process downloaded geotiffs and save to CSVs '''
    try:
        # check if results already exist
        outputname = f'files/{year}/meadow_{year}_{meadowId}'
        image_name = f'files/NDVIs/{year}/meadow_{year}_{meadowId}.tif'
        if not os.path.exists(image_name):
            return -1
        try:
            df = geotiffToDataFrame(image_name, cols)
        except:
            return -2
        df = processGeotiff(df)
        
        if not df.empty:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            all_data = df.groupby(['X', 'Y']).apply(interpolate_group).reset_index(drop=True)
            data = pd.read_csv(f'{outputname}.csv')
            all_data = pd.merge(data, all_data, on=['X', 'Y'])
            all_data.to_csv(f'{outputname}.csv', index=False)
            del all_data
        else:
            meadowId = -2  # if all_data is an empty dataframe
        
        return meadowId
    except:
        return -4
'''
meadowId = 15508   # 16973 (largest), 16247 (smallest)
prepareMeadows(meadowId)
processMeadow((meadowId)
'''

years = range(1985, 2025)
# run the first phase of geotiff downloads for NDVIs
for year in years:
    start = datetime.now()
    loadYearCollection(year)
    with Parallel(n_jobs=60, prefer="threads") as parallel:
        bandresult = parallel(delayed(prepareMeadows)(meadowId) for meadowId in allIds)
    print(f"Pre-processing of tasks for {year} completed in {datetime.now() - start}")
   
# run the processing of NDVIs and appending to original dataframes
for year in years:
    start = datetime.now()
    loadYearCollection(year)
    with Parallel(n_jobs=60, prefer="threads") as parallel:
        bandresult = parallel(delayed(processMeadow)(meadowId) for meadowId in allIds)
    print(f"Year {year} completed in {datetime.now() - start}")
