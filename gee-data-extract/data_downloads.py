# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:01:41 2025

@author: jonyegbula
"""

import os
import ee
import geemap
import pandas as pd
import warnings
import geopandas as gpd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from shapely.geometry import box
warnings.filterwarnings("ignore")

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)
warnings.filterwarnings("ignore")
folder_id = "1RpZRfWUz6b7UpZfRByWSXuu0k78BAxzz"     # characters after the "folders/" in G-drive url
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

G_driveAccess()
epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("../files/AllPossibleMeadows_2025-04-01.shp").to_crs(epsg_crs)

# download 10m USGS DEM
USGS_dem = ee.Image("USGS/3DEP/10m").select('elevation')
for idx in range(shapefile.shape[0]):
    feature = shapefile.loc[idx, :].geometry
    if feature.geom_type == 'Polygon':
        shapefile_bbox = ee.Geometry.Polygon(list(feature.exterior.coords))
    elif feature.geom_type == 'MultiPolygon':
        shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geoms))
    dem = USGS_dem.clip(shapefile_bbox)
    geemap.ee_export_image_to_drive(USGS_dem, description='usgs_10m_dem_' + str(idx), folder="files", crs=epsg_crs, region=shapefile_bbox, scale=10.2, maxPixels=1e13)

# extract bounds of sierra nevada meadows with 100m buffer for flowlines download
minx, miny, maxx, maxy = shapefile.total_bounds
merged_zones = gpd.GeoDataFrame([1], geometry=[box(minx, miny, maxx, maxy)], crs=epsg_crs)
sierra_zone = ee.Geometry.Polygon(list(merged_zones.geometry[0].exterior.coords)).buffer(100)
nv_flowline = ee.FeatureCollection("projects/sat-io/open-datasets/NHD/NHD_NV/NHDFlowline").filterBounds(sierra_zone)
ca_flowline = ee.FeatureCollection("projects/sat-io/open-datasets/NHD/NHD_CA/NHDFlowline").filterBounds(sierra_zone)

gdf_list = []
# repeat below for loop and size info but for both "nv_flowline" and "ca_flowline" so that all is merged into gdf_list
nFlows = ca_flowline.size().getInfo()   # nv_flowline.size().getInfo()
for idx in range(0, nFlows, 5000):   # GEE's limit is 5000 features
    subset = ca_flowline.toList(5000, idx)  # nv_flowline.toList(5000, idx)
    # Extract 5000 flowlines at a time, convert to FeatureCollection, then to GeoJSON, then to geodataframe
    subset_fc = ee.FeatureCollection(subset)    
    geojson = geemap.ee_to_geojson(subset_fc)   
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs=epsg_crs)
    gdf_list.append(gdf)
    if idx % 20000 == 0: print(int(idx/20000), end=' ')

# merge geodataframe list and save as shapefile
all_gdfs = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
all_gdfs.drop_duplicates(inplace=True)
all_gdfs.reset_index(drop=True, inplace=True)
all_gdfs.to_file("../files/sierra_nevada_flowlines.shp", driver="ESRI Shapefile")
