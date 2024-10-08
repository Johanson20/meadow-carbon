# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

@author: jonyegbula
"""

import os
import time
import ee
import numpy as np
import pickle
import warnings
import pandas as pd
import geopandas as gpd
import geemap
import multiprocessing
import contextlib
import rioxarray as xr
from datetime import datetime
from dateutil.relativedelta import relativedelta
from geocube.api.core import make_geocube
from shapely.geometry import box, Polygon
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)
warnings.filterwarnings("ignore")
folder_id = "1RpZRfWUz6b7UpZfRByWSXuu0k78BAxzz"     # characters after the "folders/" in G-drive url

# Authenticate and initialize python access to Google Earth Engine
# ee.Authenticate()    # only use if you've never run this on your current computer before or loss GEE access
ee.Initialize()

# Authenticate and create the PyDrive client for google drive access
gauth = GoogleAuth()
# gauth.CommandLineAuth()  # Creates local webserver and auto handles authentication (only do it once).
gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:   # Authenticate if there are no valid credentials
    gauth.CommandLineAuth()
elif gauth.access_token_expired:    # Refresh the credentials if they are expired
    gauth.Refresh()
else:   # Load the existing credentials
    gauth.Authorize()
# Save the credentials for the next run
gauth.SaveCredentialsFile("mycreds.txt")
drive = GoogleDrive(gauth)


def geotiffToCsv(input_raster, bandnames):
    # creates a dataframe of unique columns (hence combines repeating band names)
    geotiff = xr.open_rasterio(input_raster)
    df = geotiff.to_dataframe(name='value').reset_index()
    df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
    geotiff.close()
    nrows = df.shape[0]

    out_csv = pd.DataFrame()
    allBands = set(df.columns)
    nBands = len(bandnames)
    
    # There are 11 repeating bands
    for col in range(1, 12):
        values = []
        for band in [col] + list(range(14+col, nBands+1, 11)):
            if band in allBands:
                values = values + list(df[band])
            else:
                values = values + [np.nan]*nrows
        out_csv[bandnames[col-1]] = values
    
    # repeat the other columns throughout length of dataframe
    n = int(out_csv.shape[0]/nrows)
    for col in range(12, 15):
        out_csv[bandnames[col-1]] = list(df[col])*n
    out_csv['x'] = list(df['x'])*n
    out_csv['y'] = list(df['y'])*n
    
    return out_csv


def processGeotiff(df):
    nullIds = list(np.where(df['tmmx'].isnull())[0])
    df.drop(nullIds, inplace = True)
    df.columns = cols[:-7]
    NA_Ids = df.isin([np.inf, -np.inf]).any(axis=1)
    df = df[~NA_Ids]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['Date'] = pd.to_timedelta(df['Date'], unit='s') + pd.to_datetime('1970-01-01')
    df['Date'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d'))
    df['NDVI'] = (df['NIR'] - df['Red'])/(df['NIR'] + df['Red'])
    df['NDWI'] = (df['NIR'] - df['SWIR_1'])/(df['NIR'] + df['SWIR_1'])
    df['EVI'] = 2.5*(df['NIR'] - df['Red'])/(df['NIR'] + 6*df['Red'] - 7.5*df['Blue'] + 1)
    df['SAVI'] = 1.5*(df['NIR'] - df['Red'])/(df['NIR'] + df['Red'] + 0.5)
    df['BSI'] = ((df['Red'] + df['SWIR_1']) - (df['NIR'] + df['Blue']))/(df['Red'] + df['SWIR_1'] + df['NIR'] + df['Blue'])
    df['NDPI'] = (df['NIR'] - (0.56*df['Red'] + 0.44*df['SWIR_2']))/(df['NIR'] + 0.56*df['Red'] + 0.44*df['SWIR_2'])
    df['NDSI'] = (df['Green'] - df['SWIR_1'])/(df['Green'] + df['SWIR_1'])
    
    return df


def interpolate_group(group):
    group = group.drop_duplicates(subset='Date').set_index('Date').reindex(date_range)
    group['Month'] = group.index.month
    interp_cols = [col for col in group.columns if col not in ['AET', 'Annual_Precipitation', 'Month']]
    # interpolate daily values for all bands except AET (which is monthly values only)
    group.loc[:, interp_cols] = group.loc[:, interp_cols].interpolate(method='time').ffill().bfill()
    group.loc[:, ['AET', 'Annual_Precipitation']] = group.loc[:, ['Month', 'AET', 'Annual_Precipitation']].groupby('Month').ffill().bfill()
    group['Annual_Precipitation'] = group.loc[:, ['Month', 'Annual_Precipitation']].groupby('Month').first().values.sum()
    
    group['Mean_Temperature'] = (group['Maximum_temperature'].mean() + group['Minimum_temperature'].mean())/2 - 273.15
    integrals = group[group.NDSI <= 0.2]
    # snow days is defined as NDSI > 0.2; water covered is defined as non-snow days with NDWI > 0.5
    group['Snow_days'] = len(date_range) - integrals.shape[0]
    group['Wet_days'] = integrals[integrals.NDWI > 0.5].shape[0]
    integrals = integrals[cols[:6] + cols[-7:-1]].sum()
    for integral in integrals.keys():
        group['d'+integral] = integrals[integral]
    # actively growing vegetation is when NDVI >= 0.2
    resp = group.loc[group['NDVI'] >= 0.2, 'CO2.umol.m2.s']
    resp = resp - 0.367*resp - (resp - 0.367*resp)*0.094
    group.loc[group['NDVI'] >= 0.2, 'CO2.umol.m2.s'] = resp
    group['Rh'] = sum(group['CO2.umol.m2.s'])*12.01*60*60*24/1e6
    group['Snow_Flux'] = sum(group.loc[group['NDSI'] > 0.2, 'CO2.umol.m2.s'])*12.01*60*60*24/1e6
    group.drop((cols[:6] + cols[7:9] + ['AET'] + cols[-7:] + ['Month']), axis=1, inplace=True)
    
    return group.head(1)


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


def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=30)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=30)


#load ML GBM models
f = open('csv/models.pckl', 'rb')
ghg_model, agb_model, bgb_model = pickle.load(f)
f.close()
ghg_col, agb_col, bgb_col = list(ghg_model.feature_names_in_), list(agb_model.feature_names_in_), list(bgb_model.feature_names_in_)

# read in shapefile, landsat and flow accumulation data and convert shapefile to WGS '84
epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/AllPossibleMeadows_2024-09-06.shp").to_crs(epsg_crs)
# file handles need to be closed for serialization of parallel processes
allIdx = shapefile.copy()
shapefile = None
shapefile = allIdx.copy()
shapefile['crs'] = "EPSG:32611"
utm_zone10 = gpd.read_file("files/CA_UTM10.shp").to_crs(epsg_crs)
allIdx = list(gpd.overlay(shapefile, utm_zone10, how="intersection").ID)
shapefile.loc[shapefile['ID'].isin(allIdx), 'crs'] = "EPSG:32610"
# add a buffer of ~111km to Sierra Nevada
minx, miny, maxx, maxy = shapefile.total_bounds
merged_zones = gpd.GeoDataFrame([1], geometry=[box(minx, miny, maxx, maxy).buffer(1)], crs=epsg_crs)
sierra_zone = ee.Geometry.Polygon(list(merged_zones.geometry[0].exterior.coords))

flow_acc_10 = ee.Image("WWF/HydroSHEDS/15ACC").clip(sierra_zone).resample('bilinear').reproject(crs="EPSG:32610", scale=30).select('b1')
flow_acc_11 = ee.Image("WWF/HydroSHEDS/15ACC").clip(sierra_zone).resample('bilinear').reproject(crs="EPSG:32611", scale=30).select('b1')
dem = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536)
slope_10 = ee.Terrain.slope(dem).clip(sierra_zone).reproject(crs="EPSG:32610", scale=30)
slope_11 = ee.Terrain.slope(dem).clip(sierra_zone).reproject(crs="EPSG:32611", scale=30)

landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat = landsat9.merge(landsat8).merge(landsat7).merge(landsat5).filterBounds(sierra_zone).map(maskAndRename)

gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(sierra_zone).select(['tmmn', 'tmmx'])
terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterBounds(sierra_zone).select(['pr', 'aet'])
daymet = ee.ImageCollection("NASA/ORNL/DAYMET_V4").filterBounds(sierra_zone).select('swe')
cols = ['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Date', 'Minimum_temperature', 'Maximum_temperature', 'Annual_Precipitation', 'AET', 'Flow', 'Slope', 'SWE', 'X', 'Y', 'NDVI', 'NDWI', 'EVI', 'SAVI', 'BSI', 'NDPI', 'NDSI']
tasks = []

# re-run this part for each unique year
year = 2021
start_year, end_year = str(year-1)+"-10-01", str(year)+"-10-01"
date_range = pd.date_range(start=start_year, end=end_year, freq='D')[:-1]
landsat_collection = landsat.filterDate(start_year, end_year)
gridmet_10 = gridmet.filterDate(start_year, end_year).map(resample10)
gridmet_11 = gridmet.filterDate(start_year, end_year).map(resample11)
terraclimate_10 = terraclimate.filterDate(start_year, end_year.replace("-10-", "-11-")).map(resample10)
terraclimate_11 = terraclimate.filterDate(start_year, end_year.replace("-10-", "-11-")).map(resample11)
daymet_10 = daymet.filterDate(str(year)+'-04-01', str(year)+'-04-02').map(resample10).first()
daymet_11 = daymet.filterDate(str(year)+'-04-01', str(year)+'-04-02').map(resample11).first()
current_time = datetime.now().timestamp()*1000


def prepareMeadows(meadowIdx):
    # extract a single meadow and it's geometry bounds; buffer inwards to remove edge effects
    feature = shapefile.loc[meadowIdx, :]
    meadowId, mycrs = feature.ID, feature.crs
    if feature.geometry.geom_type == 'Polygon':
        if feature.Area_km2 > 0.5:
            feature.geometry = feature.geometry.simplify(0.00001)
        shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords)).buffer(-30)
    elif feature.geometry.geom_type == 'MultiPolygon':
        shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms)).buffer(-30)
    
    # convert landsat image collection over each meadow to list for iteration
    landsat_images = landsat_collection.filterBounds(shapefile_bbox)
    image_list = landsat_images.toList(landsat_images.size())
    image_result = ee.Dictionary({'image_dates': landsat_images.aggregate_array('system:time_start')}).getInfo()
    dates = [date/1000 for date in image_result['image_dates']]
    noImages = len(dates)
    if not noImages:
        print('meadowId = {}, at index: {} is too small for data extraction'.format(meadowId, meadowIdx))
        return -1
    
    # clip flow, slope and daymet to meadow's bounds
    if mycrs == "EPSG:32611":
        flow_30m = flow_acc_11.clip(shapefile_bbox).toFloat()
        slope_30m = slope_11.clip(shapefile_bbox).toFloat()
        swe_30m = daymet_11.clip(shapefile_bbox).toFloat()
    else:
        flow_30m = flow_acc_10.clip(shapefile_bbox).toFloat()
        slope_30m = slope_10.clip(shapefile_bbox).toFloat()
        swe_30m = daymet_10.clip(shapefile_bbox).toFloat()
    
    combined_image = None
    noBands = 14
    subregions = [shapefile_bbox]
    
    # iterate through each landsat image and align data types (float32)
    for idx in range(noImages):
        landsat_image = ee.Image(image_list.get(idx)).toFloat()
        # use each date in which landsat image exists to extract bands of gridmet
        start_date = relativedelta(seconds = dates[idx]) + datetime(1970, 1, 1)
        date = datetime.strftime(start_date, '%Y-%m-%d')
        next_day = (start_date + relativedelta(days=1)).strftime('%Y-%m-%d')
        target_month = date[:-2] + "01"
        next_month = (start_date + relativedelta(months=1)).replace(day=1).strftime('%Y-%m-%d')
        if mycrs == "EPSG:32611":
            gridmet_30m = gridmet_10.filterBounds(shapefile_bbox).filterDate(date, next_day).first().toFloat()
            tclimate_30m = terraclimate_10.filterBounds(shapefile_bbox).filterDate(target_month, next_month).first().toFloat()
        else:
            gridmet_30m = gridmet_11.filterBounds(shapefile_bbox).filterDate(date, next_day).first().toFloat()
            tclimate_30m = terraclimate_11.filterBounds(shapefile_bbox).filterDate(target_month, next_month).first().toFloat()
        date_band = ee.Image.constant(dates[idx]).rename('Date').toFloat()
        
        # align other satellite data with landsat and make resolution (30m)
        if idx == 0:     # extract constant values once for the same meadow
            combined_image = landsat_image.addBands([date_band, gridmet_30m, tclimate_30m, flow_30m, slope_30m, swe_30m])
        else:
            noBands += 11     # 11 total of recurring bands
            combined_image = combined_image.addBands([landsat_image, date_band, gridmet_30m, tclimate_30m])
        print(idx, end=' ')
    print()
        
    if feature.Area_km2 > 11:     # split bounds of large meadows into smaller regions to stay within limit of image downloads
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

    # either directly download images of small meadows locally or export large ones to google drive before downloading locally
    for i, subregion in enumerate(subregions):
        image_name = f'files/bands/meadow_{year}_{meadowId}_{meadowIdx}_{i}.tif'
        if noBands < 1024 and feature.Area_km2 < 5:     # (image limit = 48 MB and downloads at most 1024 bands)
            with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
                geemap.ee_export_image(combined_image.clip(subregion), filename=image_name, scale=30, crs=mycrs, region=subregion)
        elif not os.path.exists(image_name):
            geemap.ee_export_image_to_drive(combined_image.clip(subregion), description=image_name[12:-4], folder="files", crs=mycrs, region=subregion, scale=30, maxPixels=1e13)
    
    return noBands


def processMeadow(meadowCues):
    meadowIdx, noBands = meadowCues
    if noBands <= 14:
        return -1
    feature = shapefile.loc[meadowIdx, :]
    meadowId, mycrs = feature.ID, feature.crs
        
    # dataframe to store results for each meadow
    all_data = pd.DataFrame(columns=cols)
    df = pd.DataFrame()
    image_names = set()
    bandnames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Date', 'tmmn', 'tmmx', 'pr', 'AET', 'b1', 'slope', 'swe']
    subregions = 1
        
    if feature.Area_km2 > 11:     # split bounds of large meadows into subregions
        xmin, ymin, xmax, ymax = feature.geometry.bounds
        num_subregions = round(np.sqrt(feature.Area_km2/5))
        subregion_width = (xmax - xmin) / num_subregions
        subregion_height = (ymax - ymin) / num_subregions
        for i in range(num_subregions):
            for j in range(num_subregions):
                subarea = Polygon([(xmin + i*subregion_width, ymin + j*subregion_height),
                                   (xmin + (i+1)*subregion_width, ymin + j*subregion_height),
                                   (xmin + (i+1)*subregion_width, ymin + (j+1)*subregion_height),
                                   (xmin + i*subregion_width, ymin + (j+1)*subregion_height)])
                if subarea.intersects(feature.geometry):
                    subregions += 1

    # process each image subregion and bandnames order
    for i in range(subregions):
        image_name = f'files/bands/meadow_{year}_{meadowId}_{meadowIdx}_{i}.tif'
        image_names.add(image_name[12:-4])
    while noBands > 14:
        bandnames = bandnames.copy() + bandnames[:11]
        noBands -= 11
    
    etasks = tasks.copy()
    while image_names:    # read in each downloaded image, process and stack them into a dataframe
        image_name = ["files/bands/" + imagename + ".tif" for imagename in image_names][0]
        if not os.path.exists(image_name):
            etasks = [task for task in etasks if task.status()['description'] in image_names]
            isOngoing = True
            filename = ""
            while isOngoing:
                newtasks = []
                for t_k in range(len(etasks)):
                    task = etasks[t_k]
                    if task.status()['description'] in image_names and task.status()['creation_timestamp_ms'] > current_time:
                        newtasks.append(task)
                        if task.status()['state'] == 'COMPLETED':
                            filename = task.status()['description']
                            isOngoing = False
                            break
                        elif task.status()['state'] == 'FAILED':
                            isOngoing = False
                            break
                        else:
                            time.sleep(2/len(image_names))
                if not newtasks:
                    isOngoing = False
            if filename:
                file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false and title contains '{filename}'"}).GetList()
                for file in file_list:
                    if file['title'] == filename + ".tif":
                        image_name = "files/bands/" + filename + ".tif"
                        file.GetContentFile(image_name)
        try:
            image_names.remove(image_name[12:-4])
            df = geotiffToCsv(image_name, bandnames)
        except:
            continue
        df = processGeotiff(df)
        all_data = pd.concat([all_data, df])
        df = pd.DataFrame()
    print("Band data extraction done!")    
    all_data.head()
    
    if not all_data.empty:
        # select relevant columns, predict GHG and interpolate daily values
        all_data.drop_duplicates(inplace=True)
        all_data.reset_index(drop=True, inplace=True)
        all_data['CO2.umol.m2.s'] = ghg_model.predict(all_data.loc[:, ghg_col])
        all_data.loc[all_data['CO2.umol.m2.s'] < 0, 'CO2.umol.m2.s'] = 0
        all_data = all_data.groupby(['X', 'Y']).apply(interpolate_group).reset_index(drop=True)

        # Predict AGB/BGB per pixel using integrals and set negative values to zero, then convert to NEP
        all_data['HerbBio.g.m2'] = agb_model.predict(all_data.loc[:, agb_col])
        all_data['Roots.kg.m2'] = bgb_model.predict(all_data.loc[:, bgb_col])
        all_data.loc[all_data['HerbBio.g.m2'] < 0, 'HerbBio.g.m2'] = 0
        all_data.loc[all_data['Roots.kg.m2'] < 0, 'Roots.kg.m2'] = 0
        all_data['BNPP'] = all_data['Roots.kg.m2']*0.6*(0.2884*np.exp(0.046*all_data['Mean_Temperature']))*0.368*1e3
        all_data['ANPP'] = all_data['HerbBio.g.m2']*0.433
        all_data['NEP'] = all_data['ANPP'] + all_data['BNPP'] - all_data['Rh']
        print("Predictions and interpolations done!")
        
        # make geodataframe of predictions and projected coordinates as crs; convert to raster
        utm_lons, utm_lats = all_data['X'], all_data['Y']
        res = 30
        out_rasters = {0: ['AGB.tif', 'HerbBio.g.m2'], 1: ['BGB.tif', 'Roots.kg.m2'], 2: ['GHG.tif', 'Rh'], 3: ['NEP.tif', 'NEP']}
        for i in range(4):
            out_raster = f'files/Image_meadow_{year}_{meadowId}_{meadowIdx}_{out_rasters[i][0]}'
            response_col = out_rasters[i][1]
            pixel_values = all_data[response_col]
            gdf = gpd.GeoDataFrame(pixel_values, geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=mycrs.split(":")[1])
            gdf.plot(column=response_col, cmap='viridis', legend=True)
            out_grd = make_geocube(vector_data=gdf, measurements=gdf.columns.tolist()[:-1], resolution=(-res, res))
            out_grd.rio.to_raster(out_raster)
        all_data.to_csv(f'files/meadow_{year}_{meadowId}_{meadowIdx}.csv', index=False)
    else:
        print('meadowId = {}, at index = {} had no valid data rows for prediction and interpolation'.format(meadowId, meadowIdx))
        meadowIdx = -2
    
    return meadowIdx


'''
meadowIdx = 16489   # (16450), 16538 (smallest)
noBands = prepareMeadows(meadowIdx)
processMeadow((meadowIdx, noBands))
'''
if __name__ == "__main__":
    allIdx = shapefile.index
    start = datetime.now()
    with multiprocessing.Pool(processes=60) as pool:
        bandresult = pool.map(prepareMeadows, allIdx)
    
    print("Pre-processing of tasks completed!")
    
    meadowData = list(zip(allIdx, bandresult))
    tasks = ee.batch.Task.list()
    with multiprocessing.Pool(processes=60) as pool:
        result = pool.map(processMeadow, meadowData)
    
    print(datetime.now() - start)
    print(f"Year {year} completed!")
