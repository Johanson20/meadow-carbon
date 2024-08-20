# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

@author: jonyegbula
"""

import os
import ee
import numpy as np
import pickle
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

mydir = "R:\SCRATCH\jonyegbula\meadow-carbon"
os.chdir(mydir)

# Authenticate and initialize python access to Google Earth Engine
# ee.Authenticate()    # only use if you've never run this on your current computer before or loss GEE access
ee.Initialize()

ncores = multiprocessing.cpu_count()
year = 2021
start_year, end_year = str(year-1)+"-10-01", str(year)+"-10-01"
date_range = pd.date_range(start=start_year, end=end_year, freq='D')[:-1]
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']).filterDate(start_year, end_year)
landsat7_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']).filterDate(start_year, end_year)
landsat5_collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']).filterDate(start_year, end_year)
landsat9_collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']).filterDate(start_year, end_year)

flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(start_year, end_year).select(['tmmn', 'tmmx', 'pr'])
dem = ee.Image('USGS/3DEP/10m').select('elevation')
slope = ee.Terrain.slope(dem)
daymet = ee.ImageCollection("NASA/ORNL/DAYMET_V4").select('swe')
cols = ['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Minimum_temperature', 'Maximum_temperature', 'Mean_Precipitation',
        'Date', 'Flow', 'Slope', 'SWE', 'X', 'Y', 'NDVI', 'NDWI', 'EVI', 'SAVI', 'BSI', 'NDPI', 'NDSI']


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


def geotiffToCsv(input_raster, bandnames):
    # creates a dataframe of unique columns (hence combines repeating band names)
    geotiff = xr.open_rasterio(input_raster)
    df = geotiff.to_dataframe(name='value').reset_index()
    df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
    geotiff.close()
    nBands = df.shape[1] - 2
    
    out_csv = pd.DataFrame()
    ncol = len(set(bandnames))
    n = ((nBands - ncol)//10 + 1)
    
    # There are 10 (repeating) landsat, gridmet and constant value bands
    for col in range(1, 11):
        values = []
        for band in [col] + list(range(ncol+col, nBands+1, 10)):
            values = values + list(df[band])
        out_csv[bandnames[col-1]] = values
    
    # repeat the other columns throughout length of dataframe
    for col in range(11, 14):
        out_csv[bandnames[col-1]] = list(df[col])*n
    out_csv['x'] = list(df['x'])*n
    out_csv['y'] = list(df['y'])*n
    
    return out_csv


def processGeotiffCsv(df):
    nullIds =  list(np.where(df['tmmx'].isnull())[0])
    df.drop(nullIds, inplace = True)
    df.reset_index(drop=True, inplace=True)
    
    # temporary dataframe for each iteration to be appended to the overall dataframe
    df.columns = cols[:-7]
    df['Date'] = pd.to_timedelta(df['Date'], unit='s') + pd.to_datetime('1970-01-01')
    df['Date'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d'))
    df['NDVI'] = (df['NIR'] - df['Red'])/(df['NIR'] + df['Red'])
    df['NDWI'] = (df['NIR'] - df['SWIR_1'])/(df['NIR'] + df['SWIR_1'])
    df['EVI'] = 2.5*(df['NIR'] - df['Red'])/(df['NIR'] + 6*df['Red'] - 7.5*df['Blue'] + 1)
    df['SAVI'] = 1.5*(df['NIR'] - df['Red'])/(df['NIR'] + df['Red'] + 0.5)
    df['BSI'] = ((df['Red'] + df['SWIR_1']) - (df['NIR'] + df['Blue']))/(df['Red'] + df['SWIR_1'] + df['NIR'] + df['Blue'])
    df['NDPI'] = (df['NIR'] - (0.56*df['Red'] + 0.44*df['SWIR_2']))/(df['NIR'] + 0.56*df['Red'] + 0.44*df['SWIR_2'])
    df['NDSI'] = (df['Green'] - df['SWIR_1'])/(df['Green'] + df['SWIR_1'])
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    NA_Ids = df.isin([np.inf, -np.inf]).any(axis=1)
    
    return df[~NA_Ids]


def interpolate_group(group):
    group.drop_duplicates(subset='Date', inplace=True)
    group = group.set_index('Date').reindex(date_range).interpolate(method='time').ffill().bfill()
    group['Mean_Precipitation'] = group['Mean_Precipitation'].mean()
    integrals = group[group.NDSI <= 0.2][cols[:6] + cols[-7:-1]].sum()
    group.loc[group['NDSI'] > 0.2, 'EVI'] = 0
    group['Snow_days'] = len(date_range) - integrals.shape[0]
    group['Wet_days'] = group[group.NDWI > 0.5].shape[0]
    for integral in integrals.keys():
        group['d'+integral] = integrals[integral]
    group.drop((cols[:6] + cols[-7:]), axis=1, inplace=True)
    
    return group


#load ML GBM models
f = open('files/models.pckl', 'rb')
ghg_model, agb_model, bgb_model = pickle.load(f)
f.close()

landsat_collection = landsat9_collection.merge(landsat8_collection).merge(landsat7_collection).merge(landsat5_collection).map(maskAndRename)

# read in shapefile, landsat and flow accumulation data and convert shapefile to WGS '84
epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/AllPossibleMeadows_2024-07-10.shp").to_crs(epsg_crs)
utm_zone10 = gpd.read_file("files/CA_UTM10.shp").to_crs(epsg_crs)
zone10_meadows, mycrs = gpd.overlay(shapefile, utm_zone10, how="intersection"), 'EPSG:32610'
zone10_meadows['Buffer'] = zone10_meadows.to_crs(mycrs).geometry.buffer(-30)
zone10_meadows = zone10_meadows[~zone10_meadows.Buffer.is_empty]
zone10_meadows.reset_index(drop=True, inplace=True)
# utm_zone11 = gpd.read_file("files/CA_UTM11.shp").to_crs(epsg_crs)
# zone11_meadows, mycrs = gpd.overlay(shapefile, utm_zone11, how="intersection"), 'EPSG:32611' # Repeat all for zone 10 (make changes)
# zone11_meadows['Buffer'] = zone11_meadows.to_crs(mycrs).geometry.buffer(-30)
# zone11_meadows = zone11_meadows[~zone11_meadows.Buffer.is_empty]
# zone11_meadows.reset_index(drop=True, inplace=True)


def processMeadow(meadowIdx):
    start = datetime.now()
    # extract a single meadow and it's geometry bounds; buffer inwards to remove trees by edges by designated amount
    feature = zone10_meadows.loc[meadowIdx, :]
    meadowId = feature.ID
    if feature.geometry.geom_type == 'Polygon':
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
        return
    
    # clip flow and slope to meadow's bounds
    flow_band = flow_acc.clip(shapefile_bbox)
    slopeDem = slope.clip(shapefile_bbox)
    daymetv4 = daymet.filterBounds(shapefile_bbox).filterDate(str(year)+'-04-01', str(year)+'-04-02').first()
    
    # dataframe to store results for each meadow
    all_data = pd.DataFrame(columns=cols)
    df, df1 = pd.DataFrame(), pd.DataFrame()
    combined_image, residue_image = None, None
    var_col, bandnames1 = [], []
    bandnames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'tmmn', 'tmmx', 'pr', 'Date', 'b1', 'slope', 'swe']
    subregions = [shapefile_bbox]
    
    # iterate through each landsat image
    for idx in range(noImages):
        landsat_image = ee.Image(image_list.get(idx)).toFloat()
        # use each date in which landsat image exists to extract bands of gridmet
        start_date = timedelta(seconds = dates[idx]) + datetime(1970, 1, 1)
        date = datetime.strftime(start_date, '%Y-%m-%d')
        
        gridmet_filtered = gridmet.filterDate(date, (start_date + timedelta(days=1)).strftime('%Y-%m-%d')).first().clip(shapefile_bbox)
        gridmet_30m = gridmet_filtered.resample('bilinear')
        date_band = ee.Image.constant(dates[idx]).rename('Date')
        
        # align other satellite data with landsat and make resolution uniform (30m)
        if idx == 0:     # only extract once for the same meadow
            daymet_swe = daymetv4.resample('bilinear').toFloat()
            flow_30m = flow_band.resample('bilinear').toFloat()
            slope_30m = slopeDem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear')
            combined_image = landsat_image.addBands([gridmet_30m, date_band, flow_30m, slope_30m, daymet_swe])
            if noImages > 100:
                bandnames1 = bandnames.copy()
                residue_image = landsat_image.addBands([gridmet_30m, date_band, flow_30m, slope_30m, daymet_swe])
        else:
            if len(bandnames) < 1014:   # geemap can download at most 1024 (1014 + 10) bands
                bandnames = bandnames.copy() + bandnames[:10]     # 10 total of: landsat, gridmet and constant bands
                combined_image = combined_image.addBands([landsat_image, gridmet_30m, date_band])
            else:
                bandnames1 = bandnames1.copy() + bandnames1[:10]
                residue_image = residue_image.addBands([landsat_image, gridmet_30m, date_band])
        print(idx, end=' ')
        
    if feature.Area_km2 > 5:     # split bounds of large meadows into smaller regions to stay within limit of image downloads
        xmin, ymin, xmax, ymax = feature.geometry.bounds
        num_subregions = round(np.sqrt(feature.Area_km2/2))
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
    else:    # some subregions of small meadows are still large
        image_name = f'files/meadow_{year}_{meadowId}_{meadowIdx}_0.tif'
        extra_image_name = f'{image_name.split(".tif")[0]}_e.tif'
        subregion = shapefile_bbox
        
        with contextlib.redirect_stdout(None):
            geemap.ee_export_image(combined_image.clip(subregion), filename=image_name, scale=30, region=subregion, crs=mycrs)
            if len(bandnames1) > 13:
                geemap.ee_export_image(residue_image.clip(subregion), filename=extra_image_name, scale=30, region=subregion, crs=mycrs)
        
        if os.path.exists(extra_image_name):
            df1 = geotiffToCsv(extra_image_name, bandnames1)
        if os.path.exists(image_name):
            df = geotiffToCsv(image_name, bandnames)
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
    
    processedDf1 = df1.empty
    for i, subregion in enumerate(subregions):
        if df.empty:
            image_name = f'files/meadow_{year}_{meadowId}_{meadowIdx}_{i}.tif'
            extra_image_name = f'{image_name.split(".tif")[0]}_e.tif'
        
            # suppress output of downloaded images (image limit = 48 MB)
            with contextlib.redirect_stdout(None):
                geemap.ee_export_image(combined_image.clip(subregion), filename=image_name, scale=30, region=subregion, crs=mycrs)
                if len(bandnames1) > 13 and df1.empty:
                    geemap.ee_export_image(residue_image.clip(subregion), filename=extra_image_name, scale=30, region=subregion, crs=mycrs)
            try:
                df = geotiffToCsv(image_name, bandnames)
                if os.path.exists(extra_image_name):
                    df1 = geotiffToCsv(extra_image_name, bandnames1)
            except:
                continue
        df = processGeotiffCsv(df)
        all_data = pd.concat([all_data, df])
        # runs if all bands beyond 1024 are already processed
        if not processedDf1:
            df1 = processGeotiffCsv(df1)
            all_data = pd.concat([all_data, df1])
            processedDf1 = True
            df1 = pd.DataFrame()
        if not df1.empty:
            df1 = processGeotiffCsv(df1)
            all_data = pd.concat([all_data, df1])
            df1 = pd.DataFrame()
        df = pd.DataFrame()
    print("\nBand data extraction done!")    
    all_data.head()
    
    if not all_data.empty:
        # select relevant columns, predict GHG and interpolate daily values
        all_data.drop_duplicates(inplace=True)
        all_data.reset_index(drop=True, inplace=True)
        var_col = list(ghg_model.feature_names_in_)
        all_data['CO2.umol.m2.s'] = ghg_model.predict(all_data.loc[:, var_col])
        all_data = all_data.groupby(['X', 'Y']).apply(interpolate_group).reset_index(drop=True)

        # Predict AGB/BGB per pixel using integrals
        uniquePts = all_data.groupby(['X', 'Y'])
        var_col = list(agb_model.feature_names_in_)
        max_Ids = uniquePts.head(1).index
        agb_bgb = pd.DataFrame()
        agb_bgb['HerbBio.g.m2'] = agb_model.predict(all_data.loc[max_Ids, var_col])
        agb_bgb['Roots.kg.m2'] = bgb_model.predict(all_data.loc[max_Ids, var_col])
        
        # Turn negative AGB/BGB to zero and interpolate GHG    
        agb_bgb.loc[agb_bgb['HerbBio.g.m2'] < 0, 'HerbBio.g.m2'] = 0
        agb_bgb.loc[agb_bgb['Roots.kg.m2'] < 0, 'Roots.kg.m2'] = 0
        agb_bgb.describe()
        print("Predictions and interpolations done!")
        
        utm_lons, utm_lats = uniquePts['X'].first().values, uniquePts['Y'].first().values
        res = 30
        # make geodataframe of predictions and projected coordinates as crs; convert to raster
        out_rasters = {1: ['_AGB.tif', 'HerbBio.g.m2'], 2: ['_BGB.tif', 'Roots.kg.m2'], 3: ['_GHG.tif', 'CO2.umol.m2.s']}
        for i in range(1,4):
            out_raster = f'files/Image_meadow_{year}_{meadowId}_{meadowIdx}_{out_rasters[i][0]}'
            if i == 3:
                pixel_values = pd.Series(uniquePts[out_rasters[i][1]].sum().values, name=out_rasters[i][1])
            else:
                pixel_values = agb_bgb[out_rasters[i][1]]
            gdf = gpd.GeoDataFrame(pixel_values, geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=mycrs.split(":")[1])
            gdf.plot(column=out_rasters[i][1], cmap='viridis', legend=True)
            out_grd = make_geocube(vector_data=gdf, measurements=gdf.columns.tolist()[:-1], resolution=(-res, res))
            out_grd.rio.to_raster(out_raster)
    print(datetime.now() - start)


# processMeadow(96)   # 3462 (largest), 54 (smallest)
allIds = list(range(2000, 2500))    # zone10_meadows.index   # [54, 773, 3482, 55, 3493, 96, 1102, 1318, 927, 1022, 1144, 1044, 3462]
with Parallel(n_jobs=ncores-12, prefer="threads") as parallel:
    parallel(delayed(processMeadow)(meadowIdx) for meadowIdx in allIds)

shapefile = None