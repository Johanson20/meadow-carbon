# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:26:17 2026

@author: jonyegbula
"""

import os
import ee
import warnings
import numpy as np
import pandas as pd
import time
import geopandas as gpd
import contextlib
import glob
import rasterio
import rioxarray as xr
import geemap
from shapely.geometry import Polygon
from joblib import Parallel, delayed

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)

ee.Initialize()
warnings.filterwarnings("ignore")


# resample images collections whose band values are not 30m for both UTM zones
def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=30)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=30)


def generateCombinedPolarisImage(crs, shapefile_bbox):
    # clip relevant images to meadow's bounds
    if crs == "EPSG:32611":
        shall_clay = shallow_perc_clay_11.clip(shapefile_bbox)
        d_clay = deep_perc_clay_11.clip(shapefile_bbox)
        shall_silt = shallow_perc_silt_11.clip(shapefile_bbox)
        d_silt = deep_perc_silt_11.clip(shapefile_bbox)
        shall_hydra = shallow_hydra_cond_11.clip(shapefile_bbox)
        d_hydra = deep_hydra_cond_11.clip(shapefile_bbox)
        shall_res_water = shallow_res_water_11.clip(shapefile_bbox)
        d_res_water = deep_res_water_11.clip(shapefile_bbox)
        shall_sat_water = shallow_sat_water_11.clip(shapefile_bbox)
        d_sat_water = deep_sat_water_11.clip(shapefile_bbox)
        shall_org = shallow_organic_m_11.clip(shapefile_bbox)
    else:
        shall_clay = shallow_perc_clay.clip(shapefile_bbox)
        d_clay = deep_perc_clay.clip(shapefile_bbox)
        shall_silt = shallow_perc_silt.clip(shapefile_bbox)
        d_silt = deep_perc_silt.clip(shapefile_bbox)
        shall_hydra = shallow_hydra_cond.clip(shapefile_bbox)
        d_hydra = deep_hydra_cond.clip(shapefile_bbox)
        shall_res_water = shallow_res_water.clip(shapefile_bbox)
        d_res_water = deep_res_water.clip(shapefile_bbox)
        shall_sat_water = shallow_sat_water.clip(shapefile_bbox)
        d_sat_water = deep_sat_water.clip(shapefile_bbox)
        shall_org = shallow_organic_m.clip(shapefile_bbox)
    
    shall_water_content = shall_sat_water.subtract(shall_res_water)
    d_water_content = d_sat_water.subtract(d_res_water)
    combined_image = shall_org.addBands([shall_clay, d_clay, shall_silt, d_silt, shall_hydra, d_hydra, shall_water_content, d_water_content])
    
    return combined_image


def generateCombinedDEMImage(crs, shapefile_bbox):
    # clip relevant images to meadow's bounds
    if crs == "EPSG:32611":
        elev_10m = dem_11.clip(shapefile_bbox)
        slope_10m = slope_11.clip(shapefile_bbox)
    else:
        elev_10m = dem_10.clip(shapefile_bbox)
        slope_10m = slope_10.clip(shapefile_bbox)
    
    combined_image = elev_10m.addBands(slope_10m)
    return combined_image


def splitMeadowBounds(feature, makeSubRegions=True, shapefile_bbox=None, tilesplit=0):
    subregions = [shapefile_bbox] if makeSubRegions else 1
    if feature.Area_km2 > 100 or tilesplit > 0:     # split bounds of large meadows into smaller regions
        xmin, ymin, xmax, ymax = feature.geometry.bounds
        num_subregions = tilesplit if tilesplit > 0 else round(np.sqrt(feature.Area_km2/25))
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


def downloadImageBands(shapefile_bbox, imagename, feature, combined_image):
    mycrs = feature.epsgCode
    # either directly download images of small meadows locally or export large ones to google drive before downloading locally
    image_name = f'{imagename}.tif'
    if not os.path.exists(image_name):     # (image limit = 48 MB and downloads at most 1024 bands)
        with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
            geemap.ee_export_image(combined_image.clip(shapefile_bbox), filename=image_name, scale=30, crs=mycrs, region=shapefile_bbox)
            if not os.path.exists(image_name):
                time.sleep(1.1)
                with contextlib.redirect_stdout(None):  # suppress output of downloaded images 
                    geemap.ee_export_image(combined_image.clip(shapefile_bbox), filename=image_name, scale=30, crs=mycrs, region=shapefile_bbox)
    return feature.ID


# DEM (for elevation and slope) has spatial resolution of 10.2m
dem_10 = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32610", scale=10.2)
dem_11 = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32611", scale=10.2)
slope_10 = ee.Terrain.slope(dem_10)
slope_11 = ee.Terrain.slope(dem_11)

# these polaris soil datasets have 30m spatial resolution
perc_clay = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample10)
hydra_cond = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample10)
perc_silt = ee.ImageCollection('projects/sat-io/open-datasets/polaris/silt_mean').select("b1").map(resample10)
residual_water = ee.ImageCollection('projects/sat-io/open-datasets/polaris/theta_s_mean').select("b1").map(resample10)
saturated_water = ee.ImageCollection('projects/sat-io/open-datasets/polaris/theta_r_mean').select("b1").map(resample10)
organic_m = ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').select("b1").map(resample10)

shallow_perc_clay = ee.ImageCollection(perc_clay.toList(3)).mean()
deep_perc_clay = ee.Image(perc_clay.toList(6).get(3))
shallow_hydra_cond = ee.ImageCollection(hydra_cond.toList(3)).mean()
deep_hydra_cond = ee.Image(hydra_cond.toList(6).get(3))
shallow_perc_silt = ee.ImageCollection(perc_silt.toList(3)).mean()
deep_perc_silt = ee.Image(perc_silt.toList(6).get(3))
shallow_res_water = ee.ImageCollection(residual_water.toList(3)).mean()
deep_res_water = ee.Image(residual_water.toList(6).get(3))
shallow_sat_water = ee.ImageCollection(saturated_water.toList(3)).mean()
deep_sat_water = ee.Image(saturated_water.toList(6).get(3))
shallow_organic_m = ee.ImageCollection(organic_m.toList(3)).mean()

# same as above but for EPSG 32611 (above is 32610)
perc_clay_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample11)
hydra_cond_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample11)
perc_silt_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/silt_mean').select("b1").map(resample11)
residual_water_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/theta_s_mean').select("b1").map(resample11)
saturated_water_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/theta_r_mean').select("b1").map(resample11)
organic_m_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').select("b1").map(resample11)

shallow_perc_clay_11 = ee.ImageCollection(perc_clay_11.toList(3)).mean()
deep_perc_clay_11 = ee.Image(perc_clay_11.toList(6).get(3))
shallow_hydra_cond_11 = ee.ImageCollection(hydra_cond_11.toList(3)).mean()
deep_hydra_cond_11 = ee.Image(hydra_cond_11.toList(6).get(3))
shallow_perc_silt_11 = ee.ImageCollection(perc_silt_11.toList(3)).mean()
deep_perc_silt_11 = ee.Image(perc_silt_11.toList(6).get(3))
shallow_res_water_11 = ee.ImageCollection(residual_water_11.toList(3)).mean()
deep_res_water_11 = ee.Image(residual_water_11.toList(6).get(3))
shallow_sat_water_11 = ee.ImageCollection(saturated_water_11.toList(3)).mean()
deep_sat_water_11 = ee.Image(saturated_water_11.toList(6).get(3))
shallow_organic_m_11 = ee.ImageCollection(organic_m_11.toList(3)).mean()

epsg_crs = "EPSG:4326"
watershed = gpd.read_file("files/NHDPlus_Watersheds_SN_20260204.shp").to_crs(epsg_crs)
shapefile = gpd.read_file("files/SierraNevadaMeadow_V3.0.shp").to_crs(epsg_crs)
shapefile['epsgCode'] = "EPSG:32611"
utm_zone10 = gpd.read_file("files/CA_UTM10.shp").to_crs(epsg_crs)
allIds = list(gpd.overlay(shapefile, utm_zone10, how="intersection").ID)
shapefile.loc[shapefile['ID'].isin(allIds), 'epsgCode'] = "EPSG:32610"
bandnames = ['Y', 'X', 'Organic_Matter', 'Shallow_Clay', 'Deep_Clay', 'Shallow_Silt', 'Deep_Silt', 'Shallow_Hydra_Conduc', 'Deep_Hydra_Conduc', 'Shallow_Water_Content', 'Deep_Water_Content', 'geometry']
allIds = shapefile.ID


def downloadMeadowBands(meadowId):
    try:
        imageDEMname = f'files/meadow_prioritization/Meadow_DEM_{meadowId}'
        if os.path.exists(imageDEMname+".tif"):
            return [meadowId]*2
        # extract a single meadow and it's geometry bounds; buffer inwards to remove edge effects
        feature = shapefile[shapefile.ID == meadowId].iloc[0]
        mycrs = feature.epsgCode
        if feature.geometry.geom_type == 'Polygon':
            if feature.Area_km2 > 0.5:
                feature.geometry = feature.geometry.simplify(0.00001)
            shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords))
        elif feature.geometry.geom_type == 'MultiPolygon':
            shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms))
        
        # combine bands and split large meadows
        combined_polaris_image = generateCombinedPolarisImage(mycrs, shapefile_bbox)
        imagePolarisName = f'files/meadow_prioritization/Meadow_Polaris_{meadowId}'
        combined_dem_image = generateCombinedDEMImage(mycrs, shapefile_bbox)
        retrievePolarisId = downloadImageBands(shapefile_bbox, imagePolarisName, feature, combined_polaris_image)
        retrieveDEMId = downloadImageBands(shapefile_bbox, imageDEMname, feature, combined_dem_image)
        
        return [retrievePolarisId, retrieveDEMId]   # the same as meadowID
    except:
        return [-1, -1]


def geotiffsToDataFrame():
    # creates a single dataframe of all geotiffs
    all_files = [f for f in glob.glob("files/meadow_prioritization/Meadow_Polaris*.tif")]
    all_data = gpd.GeoDataFrame(columns=bandnames, crs=4326)
    for file in all_files:  # read and extract bands of each Polaris geotiff before combining all
        with rasterio.Env(CPL_LOG='ERROR'):
            geotiff = xr.open_rasterio(file)
        df = geotiff.to_dataframe(name='value').reset_index()
        geotiff.close()
        df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
        meadowId = int(file.split("_")[3][:-4])
        zone = shapefile[shapefile.ID == meadowId].iloc[0].epsgCode
        # rename columns to bandnames and convert coordinates to EPSG 4326 (lat/lon)
        df.columns = bandnames[:-1]
        utm_lons, utm_lats = df['X'], df['Y']
        pixel_values = df[bandnames[2:-1]]
        gdf = gpd.GeoDataFrame(pixel_values, geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=zone).to_crs(4326)
        all_data = pd.concat([all_data, gdf])
    
    print("Process all Polaris geotiffs!")
    all_files = [f for f in glob.glob("files/meadow_prioritization/Meadow_DEM*.tif")]
    Elev, Slope = [], []    # same process as polaris but add DEM bands
    for file in all_files:
        with rasterio.Env(CPL_LOG='ERROR'):
            geotiff = xr.open_rasterio(file)
        df = geotiff.to_dataframe(name='value').reset_index()
        geotiff.close()
        df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
        Elev.extend(df[1])
        Slope.extend(df[2])
    # add DEM columns and format coordinates to single columns
    all_data['Elevation'], all_data['Slope'] = Elev, Slope
    
    all_data['X'], all_data['Y'] = [p.x for p in all_data.geometry], [p.y for p in all_data.geometry]
    all_data = gpd.sjoin(all_data,  watershed[['HUC_12', 'geometry']], how='left', predicate='within')
    all_data.drop(['geometry', 'index_right'], axis=1, inplace=True)
    all_data.to_csv("files/meadow_prioritization/All_Meadow_rows.csv", index=False)


# downloadMeadowBands(15508)
# downloadMeadowBands(15474)

with Parallel(n_jobs=18, prefer="threads") as parallel:
    result = parallel(delayed(downloadMeadowBands)(meadowId) for meadowId in allIds)
