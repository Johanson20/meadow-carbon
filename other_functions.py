# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:45:06 2024

@author: jonyegbula
"""

import os
import pandas as pd
import geopandas as gpd
import rioxarray as xr
import contextlib
import ee
import geemap
from shapely.geometry import Polygon
from osgeo import gdal, osr
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from geocube.api.core import make_geocube


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


def geotiffToCsv(input_raster, csvfile, folderPath="", epsg=4269):
    '''
    Convert geotiff to raster and make coordinates in latlon from projected, supply epsg code of projection
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
    
    # from raster projection to 4269
    mycrs = osr.SpatialReference()
    mycrs.ImportFromProj4(geotiff.GetProjection())
    target_crs = osr.SpatialReference()
    target_crs.ImportFromEPSG(epsg)  # WGS84
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
shapefile = gpd.read_file("files/AllPossibleMeadows_2024-11-5.shp").to_crs(epsg_crs)
allIdx = shapefile.copy()
shapefile = None
shapefile = allIdx.copy()
shapefile['crs'] = "EPSG:32611"
utm_zone10 = gpd.read_file("files/CA_UTM10.shp").to_crs(epsg_crs)
allIdx = list(gpd.overlay(shapefile, utm_zone10, how="intersection").ID)
shapefile.loc[shapefile['ID'].isin(allIdx), 'crs'] = "EPSG:32610"
allIdx = None


def mergeToSingleGeotiff(inputdir, outfile, endname, variable="NEP", zone=32610, crs="4326", res=30):
    all_files = [os.path.join(inputdir, f) for f in os.listdir(inputdir) if f.endswith(endname)]
    all_data = pd.DataFrame(columns=['X', 'Y', variable])
    shps_to_use = shapefile[shapefile['crs'] == "EPSG:" + str(zone)].index
    
    if endname.endswith(".tif"):
        for file in all_files:
            if int(file.split("_")[-2]) not in shps_to_use:
                continue
            geotiff = xr.open_rasterio(file)
            df = geotiff.to_dataframe(name='value').reset_index()
            df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
            df.columns = ['X', 'Y', variable]
            geotiff.close()
            all_data = pd.concat([all_data, df])
    elif endname.endswith(".csv"):
        for file in all_files:
            if int(file.split("_")[-2]) not in shps_to_use:
                continue
            df = pd.read_csv(file)
            df = df.loc[:, ['X', 'Y', variable]]
            all_data = pd.concat([all_data, df])
    else:
        return
    
    all_data = all_data.dropna().drop_duplicates().reset_index(drop=True)
    utm_lons, utm_lats = all_data['X'], all_data['Y']
    pixel_values = all_data[variable]
    
    gdf = gpd.GeoDataFrame(pixel_values, geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=crs)
    out_grd = make_geocube(vector_data=gdf, measurements=[variable], resolution=(-res, res))
    out_grd.rio.to_raster(outfile)

# mergeToSingleGeotiff("files/2023", "files/merged_ANPP.tif", ".csv", "ANPP",)
# mergeToSingleGeotiff("files/2016NEP", "files/NEP_merged_10.tif", "_NEP.tif")
# mergeToSingleGeotiff("files/2016NEP", "files/NEP_merged_11.tif", "_NEP.tif", "NEP", 32611)