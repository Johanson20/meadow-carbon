# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

@author: Johanson C. Onyegbula
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


def G_driveAccess():
    global drive
    # Authenticate and create the PyDrive client for google drive access
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:   # Authenticate if there are no valid credentials
        gauth.LocalWebserverAuth()     # Creates local webserver and auto handles authentication (only do it once).
    elif gauth.access_token_expired:    # Refresh the credentials if they are expired
        gauth.Refresh()
    else:   # Load the existing credentials
        gauth.Authorize()
    # Save the credentials for the next run
    gauth.SaveCredentialsFile("mycreds.txt")
    drive = GoogleDrive(gauth)


def geotiffToDataFrame(input_raster, bandnames):
    # creates a dataframe of unique columns (hence combines repeating band names)
    with rasterio.Env(CPL_LOG='ERROR'):
        geotiff = xr.open_rasterio(input_raster)
    df = geotiff.to_dataframe(name='value').reset_index()
    df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
    geotiff.close()
    nrows = df.shape[0]

    out_df = pd.DataFrame()
    colBands = set(df.columns)
    nBands = len(bandnames)
    
    # There are as many repeating bands as the value of "recurringBands"
    for col in range(1, (recurringBands + 1)):
        values = []
        for band in [col] + list(range(allBands+col, nBands+1, recurringBands)):
            if band in colBands:
                values = values + list(df[band])
            else:
                values = values + [np.nan]*nrows
        out_df[bandnames[col-1]] = values
    
    # repeat the other columns throughout length of dataframe
    n = int(out_df.shape[0]/nrows)
    for col in range((recurringBands + 1), (allBands + 1)):
        out_df[bandnames[col-1]] = list(df[col])*n
    out_df['x'] = list(df['x'])*n
    out_df['y'] = list(df['y'])*n
    return out_df


def processGeotiff(df):
    # drop null columns and convert infinity to NAs
    nullIds = list(np.where(df['Maximum_temperature'].isnull())[0])
    df.drop(nullIds, inplace = True)
    df.columns = cols[:-7]
    df['AET'] *= 0.1
    logCols = ['Organic_Matter', 'Shallow_Hydra_Conduc', 'Deep_Hydra_Conduc']
    df[logCols] = df[logCols].apply(lambda col: np.power(10, col))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # landsat 7 scan lines leads to discrepancies in blank/filled values between landsat and gridmet combined days
    NA_Ids = df['Minimum_temperature'].isna()
    df = df[~NA_Ids]    # drop NAs and interpolate remaining landsat NAs due to scan line issues
    df = df.groupby(['X', 'Y']).apply(interpolate_pixel_group)
    nan_rows = df[df.isna().any(axis=1)]
    if not nan_rows.empty:
        df = df.apply(lambda row: spatial_interpolate(row, df) if row.isna().any() else row, axis=1)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # convert timestamps to dates and compute indices
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


def interpolate_pixel_group(group):
    # interpolate NA rows (landsat 7 scan lines) with nearest different date of same pixel
    return group.interpolate(method='linear', axis=0, limit_direction='both')


def spatial_interpolate(row, df):
    # interpolate NA rows (landsat 7 scan lines) based on median of other pixels on same date 
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
    group = group.drop_duplicates(subset='Date').set_index('Date').reindex(date_range)
    group['Month'] = group.index.month
    interp_cols = [col for col in group.columns if col not in ['AET', 'Annual_Precipitation', 'Month']]
    # interpolate daily values for all daily bands; and them monthly bands (AET and Precipitation)
    group.loc[:, interp_cols] = group.loc[:, interp_cols].interpolate(method='time').ffill().bfill()
    group.loc[:, ['AET', 'Annual_Precipitation']] = group.loc[:, ['Month', 'AET', 'Annual_Precipitation']].groupby('Month').ffill().bfill()
    group['Annual_Precipitation'] = group.loc[:, ['Month', 'Annual_Precipitation']].groupby('Month').first().values.sum()
    group[['Minimum_temperature', 'Maximum_temperature']] = group[['Minimum_temperature', 'Maximum_temperature']].groupby(group.index.month).transform('mean')
    
    try:
        # non-snow covered days
        growth_period = group[group.NDSI <= 0.2]
        growth_period['NDVI_Ratio'] = (growth_period['NDVI'] - min(growth_period['NDVI']))/(max(growth_period['NDVI']) - min(growth_period['NDVI']))
        # filter days before 7/15 where NDVI ratio > 0.2, and days from 7/15 where it is greater than 0.6
        group['Active_growth_days'] = sum(growth_period.loc[start_year:str(year-1)+'-12-31', 'NDVI_Ratio'] > 0.6) + sum(growth_period.loc[str(year)+'-07-15':end_year, 'NDVI_Ratio'] > 0.6) + sum(growth_period.loc[str(year)+'-01-01':str(year)+'-07-14', 'NDVI_Ratio'] > 0.2)
        # snow days is defined as NDSI > 0.2; water covered is defined as snow days with NDWI > 0.5
        group['Snow_days'] = len(date_range) - growth_period.shape[0]
        group['Wet_days'] = growth_period[growth_period.NDWI > 0.5].shape[0]
        '''# assign respiration values and SD for snow covered days fixed values
        group.loc[group['NDSI'] > 0.2, 'CO2.umol.m2.s'] = 0.6824
        group.loc[group['NDSI'] > 0.2, '1SD_CO2'] = 0.6328'''
        # calculate growing season (NDVI > 0.2) integrals
        integrals = growth_period[(growth_period.NDWI <= 0.5) & (growth_period.NDVI >= 0.2)]
        integrals = integrals[cols[:6] + cols[-7:]].sum()
        for integral in integrals.keys():
            group['d'+integral] = integrals[integral]
        # actively growing vegetation is when NDVI >= 0.2
        resp = group.loc[group['NDVI'] >= 0.2, ['CO2.umol.m2.s', '1SD_CO2']]
        resp = resp - 0.367*resp - (resp - 0.367*resp)*0.094
        group.loc[group['NDVI'] >= 0.2, ['CO2.umol.m2.s', '1SD_CO2']] = resp
        group.loc[:, ['Rh', '1SD_Rh']] = [sum(group['CO2.umol.m2.s']), sum(group['1SD_CO2'])]
        group.loc[:, ['Rh', '1SD_Rh']] *= 12.01*60*60*24/1e6
        group['Snow_Flux'] = sum(group.loc[group['NDSI'] > 0.2, 'CO2.umol.m2.s'])*12.01*60*60*24/1e6
        group.drop((cols[:6] + cols[-7:] + ['Month', 'dNDSI', 'CO2.umol.m2.s', '1SD_CO2']), axis=1, inplace=True)
        return group.head(1)
    except:
        return pd.DataFrame()


def makePredictions(df):
    # select relevant columns, predict GHG and interpolate daily values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['CO2.umol.m2.s'] = ghg_model.predict(df.loc[:, ghg_col])
    sd_ghg = ghg_84_model.predict(df.loc[:, ghg_sd_col])
    df['1SD_CO2'] = abs(sd_ghg - df['CO2.umol.m2.s'])
    df.loc[df['CO2.umol.m2.s'] < 0, 'CO2.umol.m2.s'] = 0
    df = df.groupby(['X', 'Y']).apply(interpolate_group).reset_index(drop=True)
    rh_draws = np.random.normal(df['Rh'].to_frame(), df['1SD_Rh'].to_frame(), size=(len(df['Rh']), 100))

    # Predict AGB/BGB per pixel using integrals and set negative values to zero, then convert to NEP
    df['HerbBio.g.m2'] = agb_model.predict(df.loc[:, agb_col])
    df['Roots.kg.m2'] = bgb_model.predict(df.loc[:, bgb_col])
    sd_agb = agb_84_model.predict(df.loc[:, agb_sd_col])
    df['1SD_ANPP'] = abs(sd_agb - df['HerbBio.g.m2'])
    sd_bgb = bgb_84_model.predict(df.loc[:, bgb_sd_col])
    bgb_sd = abs(sd_bgb - df['Roots.kg.m2'])
    df.loc[df['HerbBio.g.m2'] < 0, 'HerbBio.g.m2'] = 0
    df.loc[df['Roots.kg.m2'] < 0, 'Roots.kg.m2'] = 0
    df['Root_Turnover'] = (df['Roots.kg.m2']*0.49 - ((df['Roots.kg.m2']*0.49)*np.exp(-0.53)))*0.368*1000
    df['Root_Exudates'] = df['Roots.kg.m2']*1000*df['Active_growth_days']*12*1.04e-4
    df['BNPP'] = df['Root_Turnover'] + df['Root_Exudates']
    df['ANPP'] = df['HerbBio.g.m2']*0.433
    df['1SD_BNPP'] = (bgb_sd*0.49 - ((bgb_sd*0.49)*np.exp(-0.53)))*368 + bgb_sd*df['Active_growth_days']*12*0.104
    anpp_draws = np.random.normal(df['ANPP'].to_frame(), df['1SD_ANPP'].to_frame(), size=(len(df['ANPP']), 100))
    bnpp_draws = np.random.normal(df['BNPP'].to_frame(), df['1SD_BNPP'].to_frame(), size=(len(df['BNPP']), 100))
    df['NEP'] = df['ANPP'] + df['BNPP'] - df['Rh']
    df['1SD_NEP'] = pd.Series(np.std((anpp_draws + bnpp_draws - rh_draws), axis=1))
    return df
    

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


def calculateIndices(image):
    # calculate and add indices from landsat band values
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndwi = image.normalizedDifference(['NIR', 'SWIR_1']).rename('NDWI')
    evi = image.expression("2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'BLUE': image.select('Blue')}).rename('EVI')
    savi = image.expression("1.5 * ((NIR - RED) / (NIR + RED + 0.5))", {'NIR': image.select('NIR'), 'RED': image.select('Red')}).rename('SAVI')
    bsi = image.expression("((RED + SWIR_1) - (NIR + BLUE)) / (RED + SWIR_1 + NIR + BLUE)", {'RED': image.select('Red'), 'SWIR_1': image.select('SWIR_1'), 'NIR': image.select('NIR'), 'BLUE': image.select('Blue')}).rename('BSI')
    ndpi = image.expression("(NIR - ((0.56 * RED) + (0.44 * SWIR_2))) / (NIR + ((0.56 * RED) + (0.44 * SWIR_2)))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'SWIR_2': image.select('SWIR_2')}).rename('NDPI')
    return image.addBands([ndvi, ndwi, evi, savi, bsi, ndpi])


def loadYearCollection(year):
    # load GEE data for the relevant year and make folders for each year
    global date_range, start_year, end_year, landsat_5_year, landsat_collection, gridmet_10, gridmet_11, terraclimate_10, terraclimate_11, swe_10, swe_11
    start_year, end_year = str(year-1)+"-10-01", str(year)+"-10-01"
    date_range = pd.date_range(start=start_year, end=end_year, freq='D')[:-1]
    landsat_collection = landsat.filterDate(start_year, end_year)
    gridmet_10 = gridmet.filterDate(start_year, end_year).map(resample10)
    gridmet_11 = gridmet.filterDate(start_year, end_year).map(resample11)
    terraclimate_10 = terraclimate.filterDate(start_year, end_year.replace("-10-", "-11-")).map(resample10)
    terraclimate_11 = terraclimate.filterDate(start_year, end_year.replace("-10-", "-11-")).map(resample11)
    swe_10 = snow_we.filterDate(str(year)+'-04-01', str(year)+'-05-01').map(resample10).first()
    swe_11 = snow_we.filterDate(str(year)+'-04-01', str(year)+'-05-01').map(resample11).first()
    os.makedirs(f"files/{year}", exist_ok=True)
    os.makedirs(f"files/bands/{year}", exist_ok=True)


def downloadImageBands(subregions, imagename, feature, combined_image, residue_image, bandnames1):
    mycrs = feature.epsgCode
    # either directly download images of small meadows locally or export large ones to google drive before downloading locally
    for i, subregion in enumerate(subregions):
        image_name = f'{imagename}_{i}.tif'
        if feature.Area_km2 < 5:     # (image limit = 48 MB and downloads at most 1024 bands)
            with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
                geemap.ee_export_image(combined_image.clip(subregion), filename=image_name, scale=30, crs=mycrs, region=subregion)
                if not os.path.exists(image_name):
                    time.sleep(1.1)
                    with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
                        geemap.ee_export_image(combined_image.clip(subregion), filename=image_name, scale=30, crs=mycrs, region=subregion)
                if bandnames1 > allBands and os.path.exists(image_name):
                    extra_image_name = f'{imagename}_{i}_e.tif'
                    geemap.ee_export_image(residue_image.clip(subregion), filename=extra_image_name, scale=30, crs=mycrs, region=subregion)
                    if not os.path.exists(extra_image_name):
                        time.sleep(1.1)
                        with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
                            geemap.ee_export_image(residue_image.clip(subregion), filename=extra_image_name, scale=30, crs=mycrs, region=subregion)
        if not os.path.exists(image_name):    # use g-drive download if conventional one fails
            if bandnames1 > allBands and residue_image is not None:
                residue_image = residue_image.select(residue_image.bandNames().slice(allBands))
                bandnames1 -= allBands
                total_image = combined_image.addBands(residue_image)
            else:
                total_image = combined_image
            try:
                geemap.ee_export_image_to_drive(total_image.clip(subregion), description=image_name[17:-4], folder="files", crs=mycrs, region=subregion, scale=30, maxPixels=1e13)
            except:
                continue
    return bandnames1


def splitMeadowBounds(feature, makeSubRegions=True, shapefile_bbox=None, tilesplit=0):
    subregions = [shapefile_bbox] if makeSubRegions else 1
    if feature.Area_km2 > 22 or tilesplit > 0:     # split bounds of large meadows into smaller regions
        xmin, ymin, xmax, ymax = feature.geometry.bounds
        num_subregions = tilesplit if tilesplit > 0 else round(np.sqrt(feature.Area_km2/10))
        subregion_width = (xmax - xmin) / num_subregions
        subregion_height = (ymax - ymin) / num_subregions
        subregions = [] if makeSubRegions else 1
        for i in range(num_subregions):
            for j in range(num_subregions):
                subarea = Polygon([(xmin + i*subregion_width, ymin + j*subregion_height),
                                   (xmin + (i+1)*subregion_width, ymin + j*subregion_height),
                                   (xmin + (i+1)*subregion_width, ymin + (j+1)*subregion_height),
                                   (xmin + i*subregion_width, ymin + (j+1)*subregion_height)])
                if subarea.intersects(feature.geometry):
                    if makeSubRegions:
                        subregion = ee.Geometry.Rectangle(list(subarea.bounds))
                        subregions.append(subregion.intersection(shapefile_bbox))
                    else:
                        subregions += 1
    return subregions


def generateCombinedImage(crs, shapefile_bbox, image_list, dates):
    # clip relevant images to meadow's bounds
    if crs == "EPSG:32611":
        # flow_30m = flow_acc_11.clip(shapefile_bbox)
        elev_30m = dem_11.clip(shapefile_bbox)
        slope_30m = slope_11.clip(shapefile_bbox)
        swe_30m = swe_11.clip(shapefile_bbox)
        shall_clay = shallow_perc_clay_11.clip(shapefile_bbox)
        deep_clay = deep_perc_clay_11.clip(shapefile_bbox)
        shall_sand = shallow_sand_11.clip(shapefile_bbox)
        d_sand = deep_sand_11.clip(shapefile_bbox)
        shall_hydra = shallow_hydra_cond_11.clip(shapefile_bbox)
        deep_hydra = deep_hydra_cond_11.clip(shapefile_bbox)
        shall_org = shallow_organic_m_11.clip(shapefile_bbox)
        lith = lithology_11.clip(shapefile_bbox)
    else:
        # flow_30m = flow_acc_10.clip(shapefile_bbox)
        elev_30m = dem_10.clip(shapefile_bbox)
        slope_30m = slope_10.clip(shapefile_bbox)
        swe_30m = swe_10.clip(shapefile_bbox)
        shall_clay = shallow_perc_clay.clip(shapefile_bbox)
        deep_clay = deep_perc_clay.clip(shapefile_bbox)
        shall_sand = shallow_sand.clip(shapefile_bbox)
        d_sand = deep_sand.clip(shapefile_bbox)
        shall_hydra = shallow_hydra_cond.clip(shapefile_bbox)
        deep_hydra = deep_hydra_cond.clip(shapefile_bbox)
        shall_org = shallow_organic_m.clip(shapefile_bbox)
        lith = lithology.clip(shapefile_bbox)
    # filter for summer and fall and calculate indices
    landsat_5_year = landsat.filterDate(str(int(year)-6)+"-10-01", str(year-1)+"-10-01").filterBounds(shapefile_bbox).map(calculateIndices)
    landsat_June = landsat_5_year.select(['NDWI', 'EVI', 'SAVI', 'BSI', 'NDPI']).filter(ee.Filter.calendarRange(6, 6, 'month')).mean()
    landsat_Sept = landsat_5_year.select(['NDWI', 'EVI', 'SAVI', 'BSI', 'NDPI']).filter(ee.Filter.calendarRange(9, 9, 'month')).mean()
    combined_image, residue_image = None, None
    noBands, bandnames1, threshold = allBands, 0, np.ceil((1024 - allBands)/recurringBands)
    noImages = len(dates)
    
    # iterate through each landsat image and align data types (float32)
    for idx in range(noImages):
        landsat_image = ee.Image(image_list.get(idx))
        # use each date in which landsat image exists to extract bands of gridmet
        start_date = relativedelta(seconds = dates[idx]) + datetime(1970, 1, 1)
        date = datetime.strftime(start_date, '%Y-%m-%d')
        next_day = (start_date + relativedelta(days=1)).strftime('%Y-%m-%d')
        target_month = date[:-2] + "01"
        next_month = (start_date + relativedelta(months=1)).replace(day=1).strftime('%Y-%m-%d')
        if crs == "EPSG:32610":
            gridmet_30m = gridmet_10.filterBounds(shapefile_bbox).filterDate(date, next_day).first()
            tclimate_30m = terraclimate_10.filterBounds(shapefile_bbox).filterDate(target_month, next_month).first()
        else:
            gridmet_30m = gridmet_11.filterBounds(shapefile_bbox).filterDate(date, next_day).first()
            tclimate_30m = terraclimate_11.filterBounds(shapefile_bbox).filterDate(target_month, next_month).first()
        date_band = ee.Image.constant(dates[idx]).rename('Date')
        
        # align other satellite data with landsat and make resolution (30m)
        if idx == 0:     # extract constant values once for the same meadow
            combined_image = landsat_image.addBands([date_band, gridmet_30m, tclimate_30m, elev_30m, slope_30m, swe_30m, shall_clay, deep_clay, shall_sand, d_sand, shall_hydra, deep_hydra, shall_org, lith, landsat_June, landsat_Sept])
            if noImages > threshold:   # split image when bands would exceed 1024
                bandnames1 = allBands
                residue_image = landsat_image.addBands([date_band, gridmet_30m, tclimate_30m, elev_30m, slope_30m, swe_30m, shall_clay, deep_clay, shall_sand, d_sand, shall_hydra, deep_hydra, shall_org, lith, landsat_June, landsat_Sept])
        else:
            if noBands < (1024 - recurringBands):
                noBands += recurringBands     # for number of recurring bands
                combined_image = combined_image.addBands([landsat_image, date_band, gridmet_30m, tclimate_30m])
            else:
                bandnames1 += recurringBands
                residue_image = residue_image.addBands([landsat_image, date_band, gridmet_30m, tclimate_30m])
        if residue_image:
            residue_image = residue_image.toFloat()
    return [combined_image.toFloat(), residue_image, noBands, bandnames1]


def downloadFinishedTasks(image_names):
    if not 'tasks' in globals():
        tasks = ee.batch.Task.list()
    taskImages = image_names.copy()
    while taskImages:    # read in each downloaded image, process and stack them into a dataframe
        folder_id = ""
        image_name = [f"files/bands/{year}/" + imagename + ".tif" for imagename in taskImages][0]
        # check that the image isn't already downloaded and it isn't a residue image (which won't be in G_drive)
        if not os.path.exists(image_name) and not image_name.endswith('e.tif'):
            tasks = [task for task in tasks if task.config['description'] in taskImages]
            isOngoing = True
            filename = ""
            while isOngoing:
                newtasks = []
                for t_k in range(len(tasks)):
                    task = tasks[t_k]
                    if task.status()['description'] in taskImages and task.status()['creation_timestamp_ms'] > current_time:
                        newtasks.append(task)
                        if task.status()['state'] == 'COMPLETED':
                            filename = task.status()['description']
                            isOngoing = False
                            folder_id = task.status()['destination_uris'][0].split("/")[-1]
                            break
                        elif task.status()['state'] in ['FAILED', 'CANCELLED', 'CANCEL_REQUESTED']:
                            if t_k == len(tasks) - 1:   # in case a failed task is resubmitted (it loops to the other)
                                isOngoing = False
                            continue
                        else:
                            time.sleep(len(taskImages)/2)
                if not newtasks:
                    isOngoing = False
            if filename:    # load all files matching the filename from google drive)
                try:
                    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false and title contains '{filename}'"}).GetList()
                except:
                    return -3      # no filename matching the image_name is found    
                for file in file_list:
                    if file['title'] == filename + ".tif":
                        image_name = f"files/bands/{year}/{filename}.tif"
                        try:
                            file.GetContentFile(image_name)     # download file from G-drive to local folder
                        except:
                            continue
        taskImages.remove(image_name[17:-4])
    return 0


# resample and reproject when image's pixel size is not 30m for both UTM zones
def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=30)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=30)


#load ML GBM and SD models
with open('files/soil_models.pckl', 'rb') as f:
    ghg_model, agb_model, bgb_model = pickle.load(f)
with  open('files/sd_models.pckl', 'rb') as f:
    ghg_84_model, agb_84_model, bgb_84_model = pickle.load(f)
ghg_col, agb_col, bgb_col = list(ghg_model.feature_names_in_), list(agb_model.feature_names_in_), list(bgb_model.feature_names_in_)
ghg_sd_col, agb_sd_col, bgb_sd_col = list(ghg_84_model.feature_names_in_), list(agb_84_model.feature_names_in_), list(bgb_84_model.feature_names_in_)

# read in shapefile, landsat and flow accumulation data and convert shapefile to WGS '84
epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/AllPossibleMeadows_2025-07-09.shp").to_crs(epsg_crs)
# file handles need to be closed for serialization of parallel processes
shapefile['epsgCode'] = "EPSG:32611"
utm_zone10 = gpd.read_file("files/CA_UTM10.shp").to_crs(epsg_crs)
allIds = list(gpd.overlay(shapefile, utm_zone10, how="intersection").ID)
shapefile.loc[shapefile['ID'].isin(allIds), 'epsgCode'] = "EPSG:32610"
# add a buffer of 100m to Sierra Nevada
minx, miny, maxx, maxy = shapefile.total_bounds
merged_zones = gpd.GeoDataFrame([1], geometry=[box(minx, miny, maxx, maxy)], crs=epsg_crs)
sierra_zone = ee.Geometry.Polygon(list(merged_zones.geometry[0].exterior.coords)).buffer(100)

# load all relevant GEE images/collections for both UTM Zones
# flow_acc_10 = ee.Image("WWF/HydroSHEDS/15ACC").clip(sierra_zone).resample('bilinear').reproject(crs="EPSG:32610", scale=30).select('b1')
# flow_acc_11 = ee.Image("WWF/HydroSHEDS/15ACC").clip(sierra_zone).resample('bilinear').reproject(crs="EPSG:32611", scale=30).select('b1')
warnings.filterwarnings("ignore", category=DeprecationWarning)
dem_10 = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32610", scale=30)
dem_11 = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32611", scale=30)
slope_10 = ee.Terrain.slope(dem_10).clip(sierra_zone)
slope_11 = ee.Terrain.slope(dem_11).clip(sierra_zone)
'''
dem = ee.ImageCollection('USGS/3DEP/10m_collection').filterBounds(sierra_zone).mosaic().select('elevation')
dem_11 = dem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32611", scale=30)
dem_10 = dem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32610", scale=30)
slope_10 = ee.Terrain.slope(dem_10)
slope_11 = ee.Terrain.slope(dem_11)
dem_10, dem_11 = dem_10.setDefaultProjection("EPSG:4326"), dem_11.setDefaultProjection("EPSG:4326")
'''
landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat = landsat9.merge(landsat8).merge(landsat7).merge(landsat5).filterBounds(sierra_zone).map(maskAndRename)

perc_clay = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample10)
perc_sand = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').select("b1").map(resample10)
hydra_cond = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample10)
organic_m = ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').select("b1").map(resample10)
perc_clay_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample11)
perc_sand_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').select("b1").map(resample11)
hydra_cond_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample11)
organic_m_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').select("b1").map(resample11)

shallow_perc_clay_11 = ee.ImageCollection(perc_clay_11.toList(3)).mean()
deep_perc_clay_11 = ee.Image(perc_clay_11.toList(6).get(3))
shallow_sand_11 = ee.ImageCollection(perc_sand_11.toList(3)).mean()
deep_sand_11 = ee.Image(perc_sand_11.toList(6).get(3))
shallow_hydra_cond_11 = ee.ImageCollection(hydra_cond_11.toList(3)).mean()
deep_hydra_cond_11 = ee.Image(hydra_cond_11.toList(6).get(3))
shallow_organic_m_11 = ee.ImageCollection(organic_m_11.toList(3)).mean()
lithology_11 = ee.Image("CSP/ERGo/1_0/US/lithology").select("b1").resample("bilinear").reproject(crs="EPSG:32611", scale=30)
shallow_perc_clay = ee.ImageCollection(perc_clay.toList(3)).mean()
deep_perc_clay = ee.Image(perc_clay.toList(6).get(3))
shallow_sand = ee.ImageCollection(perc_sand.toList(3)).mean()
deep_sand = ee.Image(perc_sand.toList(6).get(3))
shallow_hydra_cond = ee.ImageCollection(hydra_cond.toList(3)).mean()
deep_hydra_cond = ee.Image(hydra_cond.toList(6).get(3))
shallow_organic_m = ee.ImageCollection(organic_m.toList(3)).mean()
lithology = ee.Image("CSP/ERGo/1_0/US/lithology").select("b1").resample("bilinear").reproject(crs="EPSG:32610", scale=30)

gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(sierra_zone).select(['tmmn', 'tmmx', 'srad'])
terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterBounds(sierra_zone).select(['pr', 'aet'])
snow_we = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterBounds(sierra_zone).select('swe')
cols = ['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Date', 'Minimum_temperature', 'Maximum_temperature', 'SRad', 'Annual_Precipitation', 'AET', 'Elevation', 'Slope', 'SWE', 'Shallow_Clay', 'Deep_Clay', 'Shallow_Sand', 'Deep_Sand', 'Shallow_Hydra_Conduc', 'Deep_Hydra_Conduc', 'Organic_Matter', 'Lithology', 'NDWI_June', 'EVI_June', 'SAVI_June', 'BSI_June', 'NDPI_June', 'NDWI_Sept', 'EVI_Sept', 'SAVI_Sept', 'BSI_Sept', 'NDPI_Sept', 'X', 'Y', 'NDVI', 'NDWI', 'EVI', 'SAVI', 'BSI', 'NDPI', 'NDSI']
recurringBands, allBands = len(cols[:12]), len(cols[:-9])
G_driveAccess()
allIds = shapefile.ID
current_time = datetime.strptime('07/09/2025', '%m/%d/%Y').timestamp()*1000

# re-run this part for each unique year
year = 2021
loadYearCollection(year)


def prepareMeadows(meadowId):
    try:
        # extract a single meadow and it's geometry bounds; buffer inwards to remove edge effects
        feature = shapefile[shapefile.ID == meadowId].iloc[0]
        mycrs = feature.epsgCode
        if feature.geometry.geom_type == 'Polygon':
            if feature.Area_km2 > 0.5:
                feature.geometry = feature.geometry.simplify(0.00001)
            shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords)).buffer(-30)
        elif feature.geometry.geom_type == 'MultiPolygon':
            shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms)).buffer(-30)
        
        # convert landsat image collection over each meadow to list for iteration
        landsat_images = landsat_collection.filterBounds(shapefile_bbox)
        image_list = landsat_images.toList(landsat_images.size())
        try:
            image_result = ee.Dictionary({'image_dates': landsat_images.aggregate_array('system:time_start')}).getInfo()
        except:
            time.sleep(1.1)
            try:
                image_result = ee.Dictionary({'image_dates': landsat_images.aggregate_array('system:time_start')}).getInfo()
            except:
                print(f"Meadow {meadowId} threw an exception!")
                return -3
        dates = [date/1000 for date in image_result['image_dates']]
        if not len(dates):
            return -1
        # combine bands and split large meadows
        combined_image, residue_image, noBands, bandnames1 = generateCombinedImage(mycrs, shapefile_bbox, image_list, dates)
        subregions = splitMeadowBounds(feature, True, shapefile_bbox)
        imagename = f'files/bands/{year}/meadow_{year}_{meadowId}'
        bandnames1 = downloadImageBands(subregions, imagename, feature, combined_image, residue_image, bandnames1)
        
        return noBands + bandnames1
    except:
        return -4


def processMeadow(meadowCues):
    try:
        meadowId, totalBands = meadowCues
        if totalBands <= allBands:
            return -1   # if no GEE data is available for the meadow's buffer
        feature = shapefile[shapefile.ID == meadowId].iloc[0]
        mycrs = feature.epsgCode
            
        # dataframe to store results for each meadow
        all_data = pd.DataFrame(columns=cols)
        df = pd.DataFrame()
        noBands = (1024 - recurringBands) if totalBands > 1024 else totalBands
        noBands1 = totalBands - noBands
        image_names = set()
        inputname = f'files/bands/{year}/meadow_{year}_{meadowId}'
        outputname = f'files/{year}/meadow_{year}_{meadowId}'
        bandnames = cols[:allBands]
        bandnames1 = bandnames.copy() if totalBands > 1024 else []
        subregions = splitMeadowBounds(feature, False)
        
        # process each image subregion and bandnames order
        for i in range(subregions):
            image_name = f'{inputname}_{i}.tif'
            image_names.add(image_name[17:-4])
            if totalBands > 1024 and feature.Area_km2 < 5:
                extra_image_name = f'{inputname}_{i}_e.tif'
                image_names.add(extra_image_name[17:-4])
        
        if totalBands < 1024 or not os.path.exists(f'{meadowId}_0_e.tif'):
            while totalBands > allBands:
                bandnames = bandnames.copy() + bandnames[:recurringBands]
                totalBands -= recurringBands
        else:
            while noBands > allBands:
                bandnames = bandnames.copy() + bandnames[:recurringBands]
                noBands -= recurringBands
            while noBands1 > allBands:
                bandnames1 = bandnames1.copy() + bandnames1[:recurringBands]
                noBands1 -= recurringBands
        
        if os.path.exists(f'{outputname}.csv'):
            return meadowId
        while image_names:    # read in each downloaded image, process and stack them into a dataframe
            image_name = [f"files/bands/{year}/" + imagename + ".tif" for imagename in image_names][0]
            if downloadFinishedTasks(image_names) == -3:
                return -3   # for failure in downloading completed GEE tasks
            try:
                image_names.remove(image_name[17:-4])
                if image_name.endswith('e.tif'):
                    df = geotiffToDataFrame(image_name, bandnames1)
                else:
                    df = geotiffToDataFrame(image_name, bandnames)
            except:
                continue
            df = processGeotiff(df)
            all_data = pd.concat([all_data, df])
            df = pd.DataFrame()
        all_data.head()
        
        if not all_data.empty:
            all_data = makePredictions(all_data)
            all_data[['Meadow_ID', 'Water_Year']] = meadowId, year
            # make geodataframe of predictions and projected coordinates as crs; convert to raster
            utm_lons, utm_lats = all_data['X'], all_data['Y']
            res = 30
            out_rasters = [['ANPP.tif', 'ANPP'], ['BNPP.tif', 'BNPP'], ['Rh.tif', 'Rh'], ['NEP.tif', 'NEP'], ['1SD_ANPP.tif', '1SD_ANPP'], ['1SD_BNPP.tif', '1SD_BNPP'], ['1SD_Rh.tif', '1SD_Rh'], ['1SD_NEP.tif', '1SD_NEP']]
            for i in range(8):
                response_col = out_rasters[i][1]
                pixel_values = all_data[response_col]
                gdf = gpd.GeoDataFrame(pixel_values, geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=mycrs).to_crs(3310)
                gdf.plot(column=response_col, cmap='viridis', legend=True)
                out_raster = f'{outputname}_{out_rasters[i][0]}'
                out_grd = make_geocube(vector_data=gdf, measurements=gdf.columns.tolist()[:-1], resolution=(-res, res))
                out_grd = out_grd.rio.reproject(epsg_crs)
                out_grd.rio.to_raster(out_raster)
            all_data.to_csv(f'{outputname}.csv', index=False)
        else:
            meadowId = -2  # if all_data is an empty dataframe
        
        return meadowId
    except:
        print(f"Meadow {meadowId} threw an exception!")
        return -4

'''
meadowId = 15508   # 15474 (largest), 16247 (smallest)
noBands = prepareMeadows(meadowId)
processMeadow((meadowId, noBands))
'''
if __name__ == "__main__":
    years = range(1984, 2025)
    # run the first prepareMeadows for 5 years at a time (due to GEE limit) and display progress per year
    for year in years[-6:-1]:   # modify the indexes
        start = datetime.now()
        loadYearCollection(year)
        with multiprocessing.Pool(processes=60) as pool:
            bandresult = pool.map(prepareMeadows, allIds)
        with open(f'files/{year}/bandresult.pckl', 'wb') as f:
            pickle.dump(bandresult, f)
        print(f"Pre-processing of tasks for {year} completed in {datetime.now() - start}")
    
    tasks = ee.batch.Task.list()    # load all initiated tasks
    # run the processMeadows for 5 years at a time
    for year in years[-6:-1]:   # modify the indexes
        start = datetime.now()
        loadYearCollection(year)
        with open(f'files/{year}/bandresult.pckl', 'rb') as f:
            bandresult = pickle.load(f)
        meadowData = list(zip(allIds, bandresult))
        tasks = [task for task in tasks if str(year) in task.config['description']]
        with multiprocessing.Pool(processes=60) as pool:
            result = pool.map(processMeadow, meadowData)
        with open(f'files/{year}/finalresult.pckl', 'wb') as f:
            pickle.dump(result, f)
        print(f"Year {year} completed in {datetime.now() - start}")
    
    shapefile = None