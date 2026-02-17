# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 12:19:45 2026

@author: jonyegbula
"""

import os
import ee
import warnings
import geopandas as gpd
import contextlib
import geemap

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)

ee.Initialize()
warnings.filterwarnings("ignore")


def climateAndEmsembleImages(start_date, end_date, bioclimVar="MAT", emission="SSP2-4.5", ensembleId=1):
    # function that creates difference and percentage change dual band image for ensemble and climate models
    aogcm_ensemble_mat = aogcm_ensemble_bioclim.filter(ee.Filter.eq('emission_scenario', emission)).filter(ee.Filter.date(start_date, end_date)).filter(ee.Filter.eq('bioclim_variable', bioclimVar))
    aogcm_ensemble_mat = ee.Image(aogcm_ensemble_mat.toList(aogcm_ensemble_mat.size()).get(ensembleId))
    climate_models_mat = climate_models_bioclim.filter(ee.Filter.eq('emission_scenario', emission)).filter(ee.Filter.date(start_date, end_date)).filter(ee.Filter.eq('bioclim_variable', bioclimVar)).first()
    climate_normal_mat = climate_normals_bioclim.filter(ee.Filter.date('1971-01-01', '2001-01-01')).filter(ee.Filter.eq('bioclim_variable', bioclimVar)).first()
    # calculate relevant stats after filtering for period and other variables
    climate_diff = climate_models_mat.subtract(climate_normal_mat).rename(f"{bioclimVar}_diff")
    ensemble_diff = aogcm_ensemble_mat.subtract(climate_normal_mat).rename(f"{bioclimVar}_diff")
    climate_perc_change = (climate_diff.divide(climate_normal_mat)).multiply(100).rename(f"{bioclimVar}_percChange")
    ensemble_perc_change = (ensemble_diff.divide(climate_normal_mat)).multiply(100).rename(f"{bioclimVar}_percChange")
    return (climate_diff.addBands(climate_perc_change), ensemble_diff.addBands(ensemble_perc_change))


def minMaxImages(start_date, end_date, model="ACCESS-ESM1-5", emission="SSP2-4.5"):
    # function that creates difference and percentage change dual band image for min and max temperature
    future_tmax = climate_models_tmax.filter(ee.Filter.eq('emission_scenario', emission)).filter(ee.Filter.date(start_date, end_date)).filter(ee.Filter.eq('model', model)).mean()
    future_tmin = climate_models_tmin.filter(ee.Filter.eq('emission_scenario', emission)).filter(ee.Filter.date(start_date, end_date)).filter(ee.Filter.eq('model', model)).mean()
    past_tmax = climate_models_tmax.filter(ee.Filter.eq('emission_scenario', emission)).filter(ee.Filter.date('1971-01-01', '2001-01-01')).filter(ee.Filter.eq('model', model)).mean()
    past_tmin = climate_models_tmin.filter(ee.Filter.eq('emission_scenario', emission)).filter(ee.Filter.date('1971-01-01', '2001-01-01')).filter(ee.Filter.eq('model', model)).mean()
    # calculate relevant stats after filtering for period and other variables
    tmax_diff = future_tmax.subtract(past_tmax).rename("tmax_diff")
    tmin_diff = future_tmin.subtract(past_tmin).rename("tmin_diff")
    tmax_perc_change = (tmax_diff.divide(past_tmax)).multiply(100).rename("tmax_percChange")
    tmin_perc_change = (tmin_diff.divide(past_tmin)).multiply(100).rename("tmin_percChange")
    return (tmax_diff.addBands(tmax_perc_change), tmin_diff.addBands(tmin_perc_change))


# load relevant variables (1km resolution)
climate_models_tmax = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/Climate-Models_tmax")
climate_models_tmin = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/Climate-Models_tmin")
climate_models_bioclim = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/Climate-Models_bioclim")
aogcm_ensemble_bioclim = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/AOGCM-ensemble_bioclim")
climate_normals_bioclim = ee.ImageCollection("projects/sat-io/open-datasets/CMIP6-scenarios-NA/Climate-Normals_bioclim")

# apply functions to filter by date and other parameters
climate_MAT_diff, ensemble_MAT_diff = climateAndEmsembleImages('2041-01-01', '2071-01-01')
climate_MAT_diff1, ensemble_MAT_diff1 = climateAndEmsembleImages('2041-01-01', '2071-01-01', 'MAT', 'SSP5-8.5')
climate_MAP_diff, ensemble_MAP_diff = climateAndEmsembleImages('2041-01-01', '2071-01-01', 'MAP')
climate_MAP_diff1, ensemble_MAP_diff1 = climateAndEmsembleImages('2041-01-01', '2071-01-01', 'MAP', 'SSP5-8.5')
tmax_diff, tmin_diff = minMaxImages('2041-01-01', '2071-01-01', 'ACCESS-ESM1-5', 'SSP2-4.5')
tmax_diff1, tmin_diff1 = minMaxImages('2041-01-01', '2071-01-01', 'ACCESS-ESM1-5', 'SSP5-8.5')

# read in shapefile and bounding geometry
epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/StudyExtent_20260205.shp").to_crs(epsg_crs)
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
