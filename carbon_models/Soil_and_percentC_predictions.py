# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:45:05 2026

@author: jonyegbula
"""

# This is for predicting soil and percent carbon
import os
import glob
import pandas as pd
import geopandas as gpd
import warnings
import pickle
from geocube.api.core import make_geocube

warnings.filterwarnings("ignore")

epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/AllPossibleMeadows_2025-10-22.shp").to_crs(epsg_crs)
# identify each meadow as UTM Zone 10 or 11
utm_zone10 = gpd.read_file("files/CA_UTM10.shp").to_crs(epsg_crs)
allIds = list(gpd.overlay(shapefile, utm_zone10, how="intersection").ID)

def predictSoilandPercentCarbon(years):
    ''' predict soil and carbon models of CSV level meadow data and generate geotiffs for each year '''
    # extract models for soil and percentage carbon
    with open('files/bgb_soil_models.pckl', 'rb') as f:
        percentc_model, soilc_model = pickle.load(f)
    percentc_col, soilc_col = list(percentc_model.feature_names_in_), list(soilc_model.feature_names_in_)
    mycols = ['ID','X','Y','PercentC','SoilC']
    
    for myYear in years:
        # loop through each csv file and predict soil and percentage carbon based on model (per year)
        outfile = f"files/results/{myYear}_Meadows.csv"
        all_files = [f for f in glob.glob(f"files/{myYear}/*.csv")]
        all_data = pd.DataFrame(columns=mycols)
        
        for file in all_files:
            files = [f"{file[:6]}{year}{file[10:18]}{year}{file[22:]}" for year in range(myYear-4, myYear+1)]
            dfs = [pd.read_csv(f) for f in files if os.path.exists(f)]
            all_pixels = pd.concat(dfs, ignore_index=True)
            df = (all_pixels.groupby(['X', 'Y']).mean(numeric_only=True).reset_index())
            df['ID'] = int(file.split("_")[2][:-4])
            zone = 32610 if int(file.split("_")[2][:-4]) in allIds else 32611
            df['PercentC'] = percentc_model.predict(df.loc[:, percentc_col])
            df['SoilC'] = soilc_model.predict(df.loc[:, soilc_col])
            # convert negative predictions to 0 and save predictions to another file
            df.loc[df['PercentC'] < 0, 'PercentC'] = 0
            df.loc[df['SoilC'] < 0, 'SoilC'] = 0
            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df['X'], df['Y']), crs=zone).to_crs(4326)
            df['X'], df['Y'] = [p.x for p in gdf.geometry], [p.y for p in gdf.geometry]
            df = df[mycols]
            all_data = pd.concat([all_data, df], ignore_index=True)
        all_data = all_data.dropna().reset_index(drop=True)
        all_data = all_data[mycols]
        gdf = pd.read_csv(outfile)
        gdf['ID']  = all_data['ID']
        gdf['PercentC']  = all_data['PercentC']
        gdf['SoilC']  = all_data['SoilC']
        gdf.to_csv(outfile, index=False)
        print(myYear, end='. ')
        
        # generate geotiffs for predictions at 30m resolution (same as splitCSVToGeotiffs function)
        utm_lons, utm_lats = all_data['X'], all_data['Y']
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=4326).to_crs(3310)
        mycrs = "EPSG:4326"
        for attribute in ['PercentC','SoilC']:
            gdf[attribute] = all_data[attribute]
            out_grd = make_geocube(vector_data=gdf, measurements=[attribute], resolution=(-30, 30))
            out_grd = out_grd.rio.reproject(mycrs)
            out_grd = out_grd.astype("float32").chunk({"x": 2048, "y": 2048})
            out_grd.rio.to_raster((outfile[:19] + attribute + ".tif"), tiled=True, compress="LZW", dtype="float32")
            gdf.drop(attribute, axis=1, inplace=True)
            all_data.drop(attribute, axis=1, inplace=True)
            print(attribute, "done!")

# predictSoilandPercentCarbon(list(range(1990, 2025, 5)) + [2024])
# predictSoilandPercentCarbon([2024])
