# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:45:06 2024

@author: Johanson C. Onyegbula
"""

import os
import ast
import glob
import pandas as pd
import geopandas as gpd
import rioxarray as xr
import rasterio
import contextlib
import warnings
import ee
import geemap
from shapely.geometry import Polygon
from osgeo import gdal, osr
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from geocube.api.core import make_geocube

warnings.filterwarnings("ignore")
gdal.SetConfigOption("GTIFF_SRS_SOURCE", "EPSG")
os.chdir("Code")    # change path to where github code is pulled from


def GeotiffFromGEEImage(GEE_image, bounding_box, geotiffname, bandnames, epsg, res=30):
    '''
    Creates a single geotiff file of all "bandnames" in a constrained GEE image and plots a raster of each band. Supply constrained image like clipped flow accumulation, filtered gridmet/landsat, etc. and also the ee.Geometry region it was clipped to. Include epsg code of region, spatial resolution in meters, a list of valid bandnames in the image, as well as the raster's name
    '''
    myImage = GEE_image.select(bandnames)
    geemap.ee_export_image(myImage, filename=geotiffname, scale=res, region=bounding_box, crs='EPSG:'+epsg)
    geotiff = xr.open_rasterio(geotiffname)
    df = geotiff.to_dataframe(name='value').reset_index()
    df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
    geotiff.close()
    df.columns = ['Y', 'X'] + bandnames
    
    gdf = gpd.GeoDataFrame(df[bandnames], geometry=gpd.GeoSeries.from_xy(list(df['X']), list(df['Y'])), crs=epsg)
    out_grd = make_geocube(vector_data=gdf, measurements=bandnames, resolution=(-res, res))
    out_grd.rio.to_raster(f'{geotiffname.split(".tif")[0]}_out.tif')
    for band in bandnames:
        gdf.plot(column=band, cmap='viridis', legend=True)

# GeotiffFromGEEImage(gridmet.filterBounds(shapefile_bbox).first(), shapefile_bbox, "files/landsat.tif", ['SR_B2', 'SR_B5'], '32610')


def downloadBands(banded_image, image_name, region, mycrs, scale=30):
    '''
    Split an area (if it's too large to download singly) into 4 subareas and download the bands as a geotiff, then download subareas.
    '''
    with contextlib.redirect_stdout(None):
        geemap.ee_export_image(banded_image.clip(region), filename=image_name, scale=30, region=region, crs=mycrs)
    
    if os.path.exists(image_name):
        return
    else:
        xmin, ymin, xmax, ymax = region.geometry.bounds
        num_subregions = 2
        subregion_width = (xmax - xmin) / num_subregions
        subregion_height = (ymax - ymin) / num_subregions
        count = 0
        for i in range(num_subregions):
            for j in range(num_subregions):
                subarea = Polygon([(xmin + i*subregion_width, ymin + j*subregion_height),
                                   (xmin + (i+1)*subregion_width, ymin + j*subregion_height),
                                   (xmin + (i+1)*subregion_width, ymin + (j+1)*subregion_height),
                                   (xmin + i*subregion_width, ymin + (j+1)*subregion_height)])
                if subarea.intersects(region.geometry):
                    subregion = ee.Geometry.Rectangle(list(subarea.bounds)).intersection(region)
                    new_name = f'{image_name.split(".tif")[0]}_{count}.tif'
                    downloadBands(banded_image, new_name, subregion, mycrs, scale=30)

# downloadBands(combined_image, "meadow_2021_16696_4614_0.tif", shapefile_bbox, "EPSG:32610")


def geotiffToCsv(input_raster, csvfile, folderPath="", epsg=4326):
    '''
    Convert geotiff to csv and make coordinates in latlon from projected, supply epsg code of projection
    '''
    geotiff = gdal.Open(input_raster)
    nBands = geotiff.RasterCount
    out_csv = pd.DataFrame(columns=range(nBands))
    bandnames = []
    
    for band in range(1, 1+nBands):
        bandValues = geotiff.GetRasterBand(band).ReadAsArray()
        values = []
        for value in bandValues:
            values.extend(value)
        
        bandnames.append(geotiff.GetRasterBand(band).GetDescription())
        out_csv.iloc[:, band-1] = values
    out_csv.columns = bandnames
    
    geotransform = geotiff.GetGeoTransform()
    # Loop through each pixel and extract coordinates and values
    x_geo, y_geo = [], []
    for y in range(geotiff.RasterYSize):
        for x in range(geotiff.RasterXSize):
            x_geo.append(geotransform[0] + (x+0.5) * geotransform[1] + y * geotransform[2])
            y_geo.append(geotransform[3] + x * geotransform[4] + (y+0.5) * geotransform[5])
    out_csv['x'] = x_geo
    out_csv['y'] = y_geo
    
    # from raster projection to WGS84
    mycrs = osr.SpatialReference()
    mycrs.ImportFromProj4(geotiff.GetProjection())
    target_crs = osr.SpatialReference()
    target_crs.ImportFromEPSG(epsg)
    transform = osr.CoordinateTransformation(mycrs, target_crs)
    points = [(x, y) for x, y in zip(x_geo, y_geo)]
    coords = transform.TransformPoints(points)
    out_csv['lon'] = [data[1] for data in coords]
    out_csv['lat'] = [data[0] for data in coords]
    geotiff = None
    out_csv.to_csv(csvfile, encoding='utf-8', index=False)

# geotiffToCsv("Treemap.tif", "out_xyz.csv", "", 32611)


def downloadDriveGeotiffs(nameId, delete=False, subfolder="", folder_id="1RpZRfWUz6b7UpZfRByWSXuu0k78BAxzz"):
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
    # list all files in the google drive folder with nameId in title
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false and title contains '{nameId}'"}).GetList() 
    if subfolder:
        subfolder += "/"
    if delete:
        for file in file_list:
            file.Delete()
    else:
        for file in file_list:
            image_name = f"files/bands/{subfolder}{file['title']}"
            if not os.path.exists(image_name):
                file.GetContentFile(image_name)

# downloadDriveGeotiffs('meadow_2019', False, "2019")
# downloadDriveGeotiffs('meadow_2019', True)


epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/AllPossibleMeadows_2025-07-09.shp").to_crs(epsg_crs)
# identify each meadow as UTM Zone 10 or 11
utm_zone10 = gpd.read_file("files/CA_UTM10.shp").to_crs(epsg_crs)
allIds = list(gpd.overlay(shapefile, utm_zone10, how="intersection").ID)

def mergeToSingleFile(inputdir, outfile, endname, vrt_only=True, zone=32610, res=30):
    '''This function combines all geotiffs (or csv files) of separate meadows in a specific UTM zone
    into one file (geotiff, vrt and/or csv)'''
    variable = endname.split(".")[0]
    all_data = pd.DataFrame(columns=['Y', 'X', variable])
    stats_df = pd.DataFrame(columns=['ID', 'PixelCount', 'Mean', 'Stdev', 'Min', 'Max'])
    
    if endname.endswith(".tif"):
        all_files = [f for f in glob.glob(f"{inputdir}/*{endname}") if not f.endswith(f"1SD_{endname}")]
        relevant_files = []
        for file in all_files:
            relevant_files.append(file)
            if not vrt_only:    # read all geotiffs
                # distinguish between meadows in different EPSG zones
                zone = 32610 if int(file.split("_")[2]) in allIds else 32611
                with rasterio.Env(CPL_LOG='ERROR'):
                    geotiff = xr.open_rasterio(file)
                df = geotiff.to_dataframe(name='value').reset_index()
                df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
                geotiff.close()
                if df.empty:
                    continue
                else:
                    df.columns = ['Y', 'X', variable]
                # extract summary statistics
                stats = df[variable].describe().values
                stats_df.loc[len(stats_df)] = [file.split("_")[2], stats[0]] + list(stats[2:5]) + [stats[-1]]
                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df['X'], df['Y']), crs=zone).to_crs(4326)
                df['X'], df['Y'] = [p.x for p in gdf.geometry], [p.y for p in gdf.geometry]
                all_data = pd.concat([all_data, df])
        stats_df.to_csv(outfile.split(".")[0] + "_stats.csv", index=False)
        # write to a vrt file
        vrt_path = f'{inputdir}/{variable}.vrt'
        if not os.path.exists(vrt_path):
            vrt = gdal.BuildVRT(vrt_path, relevant_files)
            vrt.FlushCache()
            vrt = None
    elif endname.endswith(".csv"):
        if "," in variable:
            mycols = ast.literal_eval(variable)
            all_data = pd.DataFrame(columns=['Y', 'X'] + mycols)
            # create summary statistics dataframe for each variable
            statCol = ['ID', 'PixelCount']
            for col in mycols:
                statCol.extend([col+"_mean", col+"_std", col+"_min", col+"_max"])
            stats_df = pd.DataFrame(columns=statCol)
        all_files = [f for f in glob.glob(f"{inputdir}/*.csv")]
        for file in all_files:
            zone = 32610 if int(file.split("_")[2][:-4]) in allIds else 32611
            df = pd.read_csv(file)
            df = df.loc[:, (['Y', 'X'] + mycols)] if "," in variable else df.loc[:, ['Y', 'X', variable]]
            if "," in variable:
                val = [file.split("_")[2][:-4], df.shape[0]]
                for col in all_data.columns[2:]:
                    stats = df[col].describe().values
                    val.extend(list(stats[2:5]) + [stats[-1]])
                stats_df.loc[len(stats_df)] = val
            else:
                stats = df[variable].describe().values
                stats_df.loc[len(stats_df)] = [file.split("_")[2], stats[0]] + list(stats[2:5]) + [stats[-1]]
            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df['X'], df['Y']), crs=zone).to_crs(4326)
            df['X'], df['Y'] = [p.x for p in gdf.geometry], [p.y for p in gdf.geometry]
            all_data = pd.concat([all_data, df])
        stats_df.to_csv(outfile.split(".")[0] + "_stats.csv", index=False)
        all_data = all_data.dropna().drop_duplicates().reset_index(drop=True)
        all_data.to_csv(outfile, index=False)
    else:
        return
    
    if not vrt_only:    # merge the geotiffs to a single image
        # create a geodataframe and then raster for the single column of interest (variable) as the pixel value
        all_data = all_data.dropna().drop_duplicates().reset_index(drop=True)
        utm_lons, utm_lats = all_data['X'], all_data['Y']
        pixel_values = all_data[variable]
        
        gdf = gpd.GeoDataFrame(pixel_values, geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=epsg_crs).to_crs(3310)
        out_grd = make_geocube(vector_data=gdf, measurements=[variable], resolution=(-res, res))
        out_grd = out_grd.rio.reproject(epsg_crs)
        out_grd = out_grd.astype("float32").chunk({"x": 2048, "y": 2048})
        out_grd.rio.to_raster(outfile, tiled=True, compress="LZW", dtype="float32")

# mergeToSingleFile("files/2023", "files/merged_BNPP.csv", "NEP.csv")
# mergeToSingleFile("files/2021", "files/2021_Meadows.csv", "['NEP','ANPP','BNPP','Rh'].csv")
# mergeToSingleFile("files/2019NEP", "files/NEP_2019_Zone10.tif", "NEP.tif")
# mergeToSingleFile("files/2019NEP", "files/NEP_2019_Zone10.tif", "NEP.tif", False)
# mergeToSingleFile("files/2016NEP", "files/NEP_2016_Zone11.tif", "1SD_NEP.tif", True, 32611)


def splitCSVToGeotiffs(inputdir, attributes=None, zone=4326, res=30):
    '''This function splits a csv file of X and Y columns alongside other variables into separate geotiffs 
    with a column represented as its own geotiff as a bandc'''
    df = pd.read_csv(inputdir)
    utm_lons, utm_lats = df['X'], df['Y']
    if not attributes:
        attributes = list(df.columns[2:])   # assumption is first 2 columns of csv are spatial coordinates
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=zone).to_crs(3310)
    mycrs = "EPSG:" + str(zone)
    
    for attribute in attributes:
        gdf[attribute] = df[attribute]
        out_grd = make_geocube(vector_data=gdf, measurements=[attribute], resolution=(-res, res))
        out_grd = out_grd.rio.reproject(mycrs)
        out_grd = out_grd.astype("float32").chunk({"x": 2048, "y": 2048})
        out_grd.rio.to_raster((inputdir[:11] + attribute + ".tif"), tiled=True, compress="LZW", dtype="float32")
        gdf.drop(attribute, axis=1, inplace=True)
        df.drop(attribute, axis=1, inplace=True)
        print(attribute, "done!")

# splitCSVToGeotiffs("files/2021_Meadows.csv", ['NEP', 'ANPP', 'BNPP', 'Rh'])


# this function was created because of a GEE glitch (their fault) that makes gee exports create new folders each time with same name
def downloadFromDuplicatedDriveFolders(nameId, myfolder="", delete=True):
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
    
    # list all folders with same name and extract with query, all files in them to download
    folders = drive.ListFile({'q': "mimeType='application/vnd.google-apps.folder' and title='files' and trashed=false"}).GetList()
    folder_ids = [f['id'] for f in folders]
    all_files = []
    
    # mini-function to go through subset of folders
    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i+n]

    for chunk in chunked(folder_ids, 500):   # extract from at most 500 folders at a time, due to limits
        folder_query = " or ".join([f"'{fid}' in parents" for fid in chunk])
        query = f"({folder_query}) and trashed=false and title contains '{nameId}'"
        files = drive.ListFile({'q': query}).GetList()
        all_files.extend(files)
    
    for file in all_files:     # download the files
        filename = file['title']
        if not os.path.exists(f"{myfolder}/{filename}"): file.GetContentFile(f"{myfolder}/{filename}")
        
        # delete file if it is the only one in folder
        if delete:
            parent_id = file['parents'][0]['id']
            remaining = drive.ListFile({'q': f"'{parent_id}' in parents and trashed=false"}).GetList()
            if len(remaining) < 2:
                folder = drive.CreateFile({'id': parent_id})
                folder.Delete()

# downloadFromDuplicatedDriveFolders("meadow_2021_", "files/bands/2021")
# downloadFromDuplicatedDriveFolders("meadow_2019", "files/bands/2019", False)
