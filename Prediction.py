# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

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


def geotiffToCsv(input_raster, bandnames):
    # creates a dataframe of unique columns (hence combines repeating band names)
    geotiff = xr.open_rasterio(input_raster)
    df = geotiff.to_dataframe(name='value').reset_index()
    df = df.pivot_table(index=['x', 'y'], columns='band', values='value').reset_index()
    geotiff.close()
    nrows = df.shape[0]

    out_csv = pd.DataFrame()
    allBands = set(df.columns)
    nBands = len(bandnames)
    
    # There are 13 repeating bands
    for col in range(1, 14):
        values = []
        for band in [col] + list(range(13+col, nBands+1, 13)):
            if band in allBands:
                values = values + list(df[band])
            else:
                values = values + [np.nan]*nrows
        out_csv[bandnames[col-1]] = values
    
    # repeat the other columns throughout length of dataframe
    n = int(out_csv.shape[0]/nrows)
    out_csv['x'] = list(df['x'])*n
    out_csv['y'] = list(df['y'])*n
    
    return out_csv


def processGeotiff(df):
    # drop null columns and convert infinity to NAs
    nullIds = list(np.where(df['SWIR_2'].isnull())[0])
    df.drop(nullIds, inplace = True)
    df.columns = cols[:-7]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # landsat 7 scan lines leads to discrepancies in NAs for landsat and gridmet
    NA_Ids = df['SWIR_2'].isna()
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
    # interpolate daily values for all bands except AET (which is monthly values only)
    group.loc[:, interp_cols] = group.loc[:, interp_cols].interpolate(method='time').ffill().bfill()
    group.loc[:, ['AET', 'Annual_Precipitation']] = group.loc[:, ['Month', 'AET', 'Annual_Precipitation']].groupby('Month').ffill().bfill()
    group['Annual_Precipitation'] = group.loc[:, ['Month', 'Annual_Precipitation']].groupby('Month').first().values.sum()
    
    group['Mean_Temperature'] = (group['Maximum_temperature'].mean() + group['Minimum_temperature'].mean())/2 - 273.15
    integrals = group[group.NDSI <= 0.2]
    # snow days is defined as NDSI > 0.2; water covered is defined as snow days with NDWI > 0.5
    group['Snow_days'] = len(date_range) - integrals.shape[0]
    group['Wet_days'] = integrals[integrals.NDWI > 0.5].shape[0]
    # actively growing vegetation is when NDVI >= 0.2
    resp = group.loc[group['NDVI'] >= 0.2, ['CO2.umol.m2.s', '1SD_CO2']]
    resp = resp - 0.367*resp - (resp - 0.367*resp)*0.094
    group.loc[group['NDVI'] >= 0.2, ['CO2.umol.m2.s', '1SD_CO2']] = resp
    group.loc[:, ['Rh', '1SD_Rh']] = [sum(group['CO2.umol.m2.s']), sum(group['1SD_CO2'])]
    group.loc[:, ['Rh', '1SD_Rh']] *= 12.01*60*60*24/1e6
    group['Snow_Flux'] = sum(group.loc[group['NDSI'] > 0.2, 'CO2.umol.m2.s'])*12.01*60*60*24/1e6
    
    return group.head(1)


def maskAndRename(image):
    image = image.rename(['Blue','Green','Red','RE_1','RE_2','RE_3','NIR','RE_4','SWIR_1','SWIR_2','SCL'])
    # mask out cloud based on bits in SCL band
    scl = image.select('SCL')
    cloud_mask = scl.neq(3).And(scl.lt(8))
    image = image.updateMask(cloud_mask).select(['Blue','Green','Red','RE_1','RE_2','RE_3','NIR','RE_4','SWIR_1','SWIR_2'])
    scaled_bands = image.multiply(1e-4)
    return image.addBands(scaled_bands, overwrite=True)


def loadYearCollection(year):
    # load GEE data for the relevant year and make folders for each year
    global date_range, sentinel1, sentinel2_10, sentinel2_11
    start_year, end_year = str(year-1)+"-10-01", str(year)+"-10-01"
    date_range = pd.date_range(start=start_year, end=end_year, freq='D')[:-1]
    sentinel1 = sentinel1.filterDate(start_year, end_year)
    sentinel2_10 = sentinel2_10.filterDate(start_year, end_year)
    sentinel2_11 = sentinel2_11.filterDate(start_year, end_year)
    os.makedirs(f"files/{year}", exist_ok=True)
    os.makedirs(f"files/bands/{year}", exist_ok=True)
    

def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=10)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=10)


#load ML GBM models
f = open('csv/models.pckl', 'rb')
ghg_model, agb_model, bgb_model = pickle.load(f)
f.close()

# read in shapefile, landsat and flow accumulation data and convert shapefile to WGS '84
epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/AllPossibleMeadows_2024-11-5.shp").to_crs(epsg_crs)
# file handles need to be closed for serialization of parallel processes
allIdx = shapefile.copy()
shapefile = None
shapefile = allIdx.copy()
shapefile['crs'] = "EPSG:32611"
utm_zone10 = gpd.read_file("files/CA_UTM10.shp").to_crs(epsg_crs)
allIdx = list(gpd.overlay(shapefile, utm_zone10, how="intersection").ID)
shapefile.loc[shapefile['ID'].isin(allIdx), 'crs'] = "EPSG:32610"
# add a buffer of ~111km (1 latitude) to Sierra Nevada
minx, miny, maxx, maxy = shapefile.total_bounds
merged_zones = gpd.GeoDataFrame([1], geometry=[box(minx, miny, maxx, maxy).buffer(1)], crs=epsg_crs)
sierra_zone = ee.Geometry.Polygon(list(merged_zones.geometry[0].exterior.coords))


cols = ['VH', 'VV', 'Date','Blue','Green','Red','RE_1','RE_2','RE_3','NIR','RE_4','SWIR_1','SWIR_2', 'X', 'Y', 'NDVI', 'NDWI', 'EVI', 'SAVI', 'BSI', 'NDPI', 'NDSI']
G_driveAccess()
allIdx = shapefile.index
current_time = datetime.strptime('20/11/2024', '%d/%m/%Y').timestamp()*1000

sentinel1 = ee.ImageCollection("COPERNICUS/S1_GRD").select(['VH', 'VV'])
sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','SCL'])
sentinel2_10 = sentinel2.map(resample10).map(maskAndRename)
sentinel2_11 = sentinel2.map(resample11).map(maskAndRename)

# re-run this part for each unique year
year = 2018
loadYearCollection(year)

def prepareMeadows(meadowIdx):
    try:
        # extract a single meadow and it's geometry bounds; buffer inwards to remove edge effects
        feature = shapefile.loc[meadowIdx, :]
        meadowId, mycrs = int(feature.ID), feature.crs
        if feature.geometry.geom_type == 'Polygon':
            if feature.Area_km2 > 0.5:
                feature.geometry = feature.geometry.simplify(0.00001)
            shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords)).buffer(-30)
        elif feature.geometry.geom_type == 'MultiPolygon':
            shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms)).buffer(-30)
        
        sentinel1_images = sentinel1.filterBounds(shapefile_bbox)
        image_list = sentinel1_images.toList(sentinel1_images.size())
        try:
            image_result = ee.Dictionary({'image_dates': sentinel1_images.aggregate_array('system:time_start')}).getInfo()
        except:
            print(f"Meadow {meadowId} threw an exception!")
            return -3
        dates = [date/1000 for date in image_result['image_dates']]
        noImages = len(dates)
        if not noImages:
            return -1
        
        combined_image, residue_image = None, None
        noBands, bandnames1 = 13, 0
        subregions = [shapefile_bbox]
        
        if mycrs == "EPSG:32611":
            sentinel2_images = sentinel2_11.filterBounds(shapefile_bbox)
        else:
            sentinel2_images = sentinel2_10.filterBounds(shapefile_bbox)
        sent_list = sentinel2_images.toList(sentinel2_images.size())
        sent_result = ee.Dictionary({'image_dates': sentinel2_images.aggregate_array('system:time_start')}).getInfo()
        sent_dates = [date/1000 for date in sent_result['image_dates']]
        
        # iterate through each landsat image and align data types (float32)
        for idx in range(noImages):
            date = dates[idx]
            c_idx = min(range(len(sent_dates)), key=lambda i: abs(sent_dates[i] - date))
            sent = ee.Image(sent_list.get(c_idx)).toFloat()
            banded_image = ee.Image(image_list.get(idx)).toFloat()
            date_band = ee.Image.constant(date).rename('Date').toFloat()        
            # align other satellite data with landsat and make resolution (30m)
            if idx == 0:     # extract constant values once for the same meadow
                combined_image = banded_image.addBands([date_band, sent])
                if noImages > 78:   # split image when bands would exceed 1024
                    bandnames1 = 13
                    residue_image = banded_image.addBands([date_band, sent])
            else:
                if noBands < 1015:
                    noBands += 13     # 11 total of recurring bands
                    combined_image = combined_image.addBands([banded_image, date_band, sent])
                else:
                    bandnames1 += 13
                    residue_image = residue_image.addBands([banded_image, date_band, sent])
        
        if feature.Area_km2 > 22:     # split bounds of large meadows into smaller regions to stay within limit of image downloads
            xmin, ymin, xmax, ymax = feature.geometry.bounds
            num_subregions = round(np.sqrt(feature.Area_km2/10))
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
            image_name = f'files/bands/{year}/meadow_{year}_{meadowId}_{meadowIdx}_{i}.tif'
            if feature.Area_km2 < 5:     # (image limit = 48 MB and downloads at most 1024 bands)
                with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
                    geemap.ee_export_image(combined_image.clip(subregion), filename=image_name, scale=10, crs=mycrs, region=subregion)
                    if not os.path.exists(image_name):
                        time.sleep(1.5)
                        with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
                            geemap.ee_export_image(combined_image.clip(subregion), filename=image_name, scale=30, crs=mycrs, region=subregion)
                    if bandnames1 > 13 and os.path.exists(image_name):
                        extra_image_name = f'{image_name.split(".tif")[0]}_e.tif'
                        geemap.ee_export_image(residue_image.clip(subregion), filename=extra_image_name, scale=30, crs=mycrs, region=subregion)
                        if not os.path.exists(extra_image_name):
                            time.sleep(1.5)
                            with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
                                geemap.ee_export_image(residue_image.clip(subregion), filename=extra_image_name, scale=30, crs=mycrs, region=subregion)
            if not os.path.exists(image_name):    # merge both images for g-drive download
                if bandnames1 > 13 and residue_image is not None:
                    total_image = combined_image.addBands(residue_image)
                else:
                    total_image = combined_image
                try:
                    geemap.ee_export_image_to_drive(total_image.clip(subregion), description=image_name[17:-4], folder="files", crs=mycrs, region=subregion, scale=30, maxPixels=1e13)
                except:
                    continue
        
        return noBands + bandnames1
    except:
        return -4


totalBands = prepareMeadows(meadowIdx)
processMeadow((meadowIdx, totalBands))

def processMeadow(meadowCues):
    try:
        meadowIdx, totalBands = meadowCues
        feature = shapefile.loc[meadowIdx, :]
        meadowId, mycrs = int(feature.ID), feature.crs
            
        # dataframe to store results for each meadow
        all_data = pd.DataFrame(columns=cols)
        df = pd.DataFrame()
        noBands = 1014 if totalBands > 1024 else totalBands
        noBands1 = totalBands - noBands
        image_names = set()
        bandnames = ['VH','VV','Date','Blue','Green','Red','RE_1','RE_2','RE_3','NIR','RE_4','SWIR_1','SWIR_2']
        bandnames1 = bandnames.copy() if totalBands > 1024 else []
        subregions = 1
            
        if feature.Area_km2 > 22:     # split bounds of large meadows into subregions
            xmin, ymin, xmax, ymax = feature.geometry.bounds
            num_subregions = round(np.sqrt(feature.Area_km2/10))
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
            image_name = f'files/bands/{year}/meadow_{year}_{meadowId}_{meadowIdx}_{i}.tif'
            image_names.add(image_name[17:-4])
            if totalBands > 1024:
                extra_image_name = f'{image_name.split(".tif")[0]}_e.tif'
                image_names.add(extra_image_name[17:-4])
        
        if totalBands < 1024:
            while totalBands > 14:
                bandnames = bandnames.copy() + bandnames[:13]
                totalBands -= 13
        else:
            while noBands > 14:
                bandnames = bandnames.copy() + bandnames[:13]
                noBands -= 13
            while noBands1 > 14:
                bandnames1 = bandnames1.copy() + bandnames1[:13]
                noBands1 -= 13
        
        tasks = ee.batch.Task.list()
        while image_names:    # read in each downloaded image, process and stack them into a dataframe
            image_name = [f"files/bands/{year}/" + imagename + ".tif" for imagename in image_names][0]
            # check that the image isn't already downloaded and it isn't a residue image (which won't be in G_drive)
            if not os.path.exists(image_name) and not image_name.endswith('e.tif'):
                tasks = [task for task in tasks if task.config['description'] in image_names]
                isOngoing = True
                filename = ""
                while isOngoing:
                    newtasks = []
                    for t_k in range(len(tasks)):
                        task = tasks[t_k]
                        if task.status()['description'] in image_names and task.status()['creation_timestamp_ms'] > current_time:
                            newtasks.append(task)
                            if task.status()['state'] == 'COMPLETED':
                                filename = task.status()['description']
                                isOngoing = False
                                break
                            elif task.status()['state'] == 'FAILED':
                                if t_k == len(tasks) - 1:   # in case a failed task is resubmitted (it loops to the other)
                                    isOngoing = False
                                continue
                            else:
                                time.sleep(2/len(image_names))
                    if not newtasks:
                        isOngoing = False
                if filename:    # load all files matching the filename from google drive)
                    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false and title contains '{filename}'"}).GetList()
                    for file in file_list:
                        if file['title'] == filename + ".tif":
                            image_name = f"files/bands/{year}/{filename}.tif"
                            file.GetContentFile(image_name)     # download file from G-drive to local folder
            try:
                image_names.remove(image_name[17:-4])
                if image_name.endswith('e.tif'):
                    df = geotiffToCsv(image_name, bandnames1)
                else:
                    df = geotiffToCsv(image_name, bandnames)
            except:
                continue
            df = processGeotiff(df)
            all_data = pd.concat([all_data, df])
            df = pd.DataFrame()
        all_data.head()
        
        if not all_data.empty:
            # select relevant columns, predict GHG and interpolate daily values
            all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            all_data.dropna(inplace=True)
            all_data.drop_duplicates(inplace=True)
            all_data.reset_index(drop=True, inplace=True)
            all_data['Roots.kg.m2'] = bgb_model.predict(all_data.loc[:, bgb_col])
            temp = gridmet.filterBounds(shapefile_bbox).filterDate('2020-10-01', '2021-10-01').mean()
            
            mean_temp = np.mean(list(temp.reduceRegion(ee.Reducer.mean(), shapefile_bbox, 30).getInfo().values())) - 273.15
            all_data['BNPP'] = all_data['Roots.kg.m2']*0.6*(0.2884*np.exp(0.046*mean_temp))*0.368*1e3
            
            x = pd.read_csv(f'files/{year}_GBM/meadow_{year}_{meadowId}_{meadowIdx}.csv')
            group = pd.merge(all_data, x, on=['X','Y'])
            all_data = group.copy()
            all_data.columns = list(all_data.columns)[:-4] + ['BNPP'] + list(all_data.columns)[-3:]
            all_data['NEP'] = all_data['ANPP'] + all_data['BNPP'] - all_data['Rh']
            
            # make geodataframe of predictions and projected coordinates as crs; convert to raster
            utm_lons, utm_lats = all_data['X'], all_data['Y']
            res = 30
            out_rasters = [['ANPP.tif', 'ANPP'], ['BNPP.tif', 'BNPP'], ['Rh.tif', 'Rh'], ['NEP.tif', 'NEP']]
            for i in range(4):
                out_raster = f'files/{year}/Image_meadow_{year}_{meadowId}_{meadowIdx}_{out_rasters[i][0]}'
                response_col = out_rasters[i][1]
                pixel_values = all_data[response_col]
                gdf = gpd.GeoDataFrame(pixel_values, geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=mycrs.split(":")[1])
                out_grd = make_geocube(vector_data=gdf, measurements=gdf.columns.tolist()[:-1], resolution=(-res, res))
                out_grd.rio.to_raster(out_raster)
                gdf.plot(column=response_col, cmap='viridis', legend=True)
            all_data.to_csv(f'files/{year}/meadow_{year}_{meadowId}_{meadowIdx}.csv', index=False)
        else:
            meadowIdx = -2
        
        return meadowIdx
    except:
        return -4

'''
meadowIdx = 16461   # (16422), 16510 (smallest)
noBands = prepareMeadows(meadowIdx)
processMeadow((meadowIdx, noBands))
'''
if __name__ == "__main__":
    years = range(1984, 2024)
    for year in years[-6:-1]:
        start = datetime.now()
        loadYearCollection(year)
        with multiprocessing.Pool(processes=60) as pool:
            bandresult = pool.map(prepareMeadows, allIdx)
        with open(f'files/{year}/bandresult.pckl', 'wb') as f:
            pickle.dump(bandresult, f)
        print(f"Pre-processing of tasks for {year} completed in {datetime.now() - start}")
    
    G_driveAccess()
    ee.Initialize()
    tasks = ee.batch.Task.list()
    for year in years[-6:-1]:
        start = datetime.now()
        loadYearCollection(year)
        with open(f'files/{year}/bandresult.pckl', 'rb') as f:
            bandresult = pickle.load(f)
        meadowData = list(zip(allIdx, bandresult))
        tasks = [task for task in tasks if str(year) in task.config['description']]
        with multiprocessing.Pool(processes=60) as pool:
            result = pool.map(processMeadow, meadowData)
        with open(f'files/{year}/finalresult.pckl', 'wb') as f:
            pickle.dump(result, f)
        print(f"Year {year} completed in {datetime.now() - start}")
