# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 12:19:45 2026

@author: jonyegbula
"""

import os
import ee
import warnings
import geopandas as gpd
import rasterio
import rioxarray as xr
import contextlib
import geemap
from shapely.geometry import Point
from geocube.api.core import make_geocube

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)

ee.Initialize()
warnings.filterwarnings("ignore")


def climateAndEmsembleImages(start_date, end_date, bioclimVar="MAT", emission="SSP2-4.5", ensembleId=1):
    # function that creates difference and percentage change dual band image for ensemble and climate models
    aogcm_ensemble_mat = aogcm_ensemble_bioclim.filter(ee.Filter.eq('emission_scenario', emission)).filter(ee.Filter.date(start_date, end_date)).filter(ee.Filter.eq('bioclim_variable', bioclimVar))
    aogcm_ensemble_mat = ee.Image(aogcm_ensemble_mat.toList(aogcm_ensemble_mat.size()).get(ensembleId))
    climate_normal_mat = climate_normals_bioclim.filter(ee.Filter.date('1971-01-01', '2001-01-01')).filter(ee.Filter.eq('bioclim_variable', bioclimVar)).first()
    # calculate relevant stats after filtering for period and other variables
    return aogcm_ensemble_mat.subtract(climate_normal_mat)


def minMaxTempImages(start_date, end_date, emission="SSP2-4.5"):
    # function that creates difference and percentage change dual band image for min and max temperature
    future_tmax = ensemble_tmax.filter(ee.Filter.eq('emission_scenario', emission)).filter(ee.Filter.date(start_date, end_date))
    # images 12 to 23 fit the date range (first 12 images and last 12 images are different dates)
    future_tmax = ee.ImageCollection(future_tmax.toList(future_tmax.size()).slice(12,24)).mean()
    future_tmin = ensemble_tmin.filter(ee.Filter.eq('emission_scenario', emission)).filter(ee.Filter.date(start_date, end_date))
    future_tmin = ee.ImageCollection(future_tmin.toList(future_tmin.size()).slice(12,24)).mean()
    past_tmax = climate_normals_tmax.filter(ee.Filter.date('1971-01-01', '2001-01-01'))
    
    # first 12 images only fit the date range, hence slicing for those and averaging
    past_tmax = ee.ImageCollection(past_tmax.toList(past_tmax.size()).slice(0,12)).mean()
    past_tmin = climate_normals_tmin.filter(ee.Filter.date('1971-01-01', '2001-01-01'))
    past_tmin = ee.ImageCollection(past_tmin.toList(past_tmin.size()).slice(0,12)).mean()
    # calculate relevant stats after filtering for period and other variables
    tmax_diff = future_tmax.subtract(past_tmax).rename("tmax_diff")
    tmin_diff = future_tmin.subtract(past_tmin).rename("tmin_diff")
    return (tmax_diff, tmin_diff)


# load relevant variables (1km resolution)
ensemble_tmax = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/AOGCM-ensemble_tmax")
ensemble_tmin = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/AOGCM-ensemble_tmin")
climate_normals_tmax = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/Climate-Normals_tmax")
climate_normals_tmin = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/Climate-Normals_tmin")
aogcm_ensemble_bioclim = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/AOGCM-ensemble_bioclim")
climate_normals_bioclim = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/Climate-Normals_bioclim")

# apply functions to filter by date and other parameters
ensemble_MAT_diff = climateAndEmsembleImages('2041-01-01', '2071-01-01')
ensemble_MAT_diff1 = climateAndEmsembleImages('2041-01-01', '2071-01-01', 'MAT', 'SSP5-8.5')
ensemble_MAP_diff = climateAndEmsembleImages('2041-01-01', '2071-01-01', 'MAP')
ensemble_MAP_diff1 = climateAndEmsembleImages('2041-01-01', '2071-01-01', 'MAP', 'SSP5-8.5')
tmax_diff, tmin_diff = minMaxTempImages('2041-01-01', '2071-01-01', 'SSP2-4.5')
tmax_diff1, tmin_diff1 = minMaxTempImages('2041-01-01', '2071-01-01', 'SSP5-8.5')

# read in shapefile and bounding geometry
epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/StudyExtent_20260331.shp").to_crs(epsg_crs)
shapefile_bbox = ee.Geometry.Polygon(list(shapefile.iloc[0].geometry.exterior.coords))

# download all geotiffs of differences in 30-year averaged values
allvars = list(globals().keys())
with contextlib.redirect_stdout(None):  # suppress output of downloaded images
    for myvar in allvars:
        if myvar.endswith("diff"):
            imagename = f'files/{myvar}_SSP2-4.5.tif'
        elif myvar.endswith("diff1"):
            imagename = f'files/{myvar[:-1]}_SSP5-8.5.tif'
        else: continue
        geemap.ee_export_image(globals()[myvar].clip(shapefile_bbox), filename=imagename, scale=1000, crs=epsg_crs, region=shapefile_bbox)


# function to read single band image
def readBand(image_name):
    with rasterio.Env(CPL_LOG='ERROR'):
        geotiff = xr.open_rasterio(image_name)
    df = geotiff.to_dataframe(name='value').reset_index()
    df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
    geotiff.close()
    return df

image1 = "files/SWE_data/swe_30y_ens32max_historical_1961-1990_1961-04-01.tif"
image2 = "files/SWE_data/swe_30y_ens32max_rcp45_2070-2099_2070-04-01.tif"
image3 = "files/SWE_data/swe_30y_ens32max_rcp45_2035-2064_2035-04-01.tif"

df1 = readBand(image1)
df2 = readBand(image2)
df = df1.drop(1, axis=1).copy()
df['abs_diff'] = df2[1] - df1[1]
df['perc_diff'] = ((df2[1] - df1[1])/df1[1])*100

geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)
gdf.set_crs(epsg_crs, inplace=True)
gdf = gpd.sjoin(gdf, shapefile, predicate='within', how='inner').drop(['x', 'y', 'index_right', 'FID'], axis=1)
gdf.to_crs(32611, inplace=True)
for col in gdf.columns[:-1]:
    out_grd = make_geocube(vector_data=gdf, measurements=[col], resolution=(-1000, 1000))
    out_grd = out_grd.rio.reproject(epsg_crs)
    out_grd.rio.to_raster(f'files/SWE_{col}_lateCentury.tif')
    # out_grd.rio.to_raster(f'files/SWE_{col}_midCentury.tif')     # if image3 is used instead of image2
