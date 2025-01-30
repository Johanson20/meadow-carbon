# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:59:46 2025

@author: jonyegbula
"""

import os
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
import ee

# Authenticate and initialize python access to Google Earth Engine
# ee.Authenticate()    # only use if you've never run this on your current computer before or loss GEE access
ee.Initialize()
mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)
warnings.filterwarnings("ignore")

# read in shapefiles, convert to WGS '84 and filter burns to 1984 (landsat beginning)
epsg_crs = "EPSG:4326"
burn_shapefile = gpd.read_file("files/California_Fire_Perimeters__all_.shp").to_crs(epsg_crs)
burn_shapefile = burn_shapefile[burn_shapefile.YEAR_ >= '1984'].reset_index(drop=True)
shapefile = gpd.read_file("files/AllPossibleMeadows_2024-11-5.shp").to_crs(epsg_crs)

# initialize final dataframe
burn_results = pd.DataFrame(columns = ["ID", "Burn_year", "Sides"])
count = 0
for meadowIdx in range(shapefile.shape[0]):
    # loop through each meadow to find out if it burned in the burn shapefile
    meadow = shapefile.loc[meadowIdx,:]
    area_burned = meadow.geometry
    x = burn_shapefile.contains(area_burned)
    # if meadow is contained completely in burn geometry, all sides burned
    if sum(x) > 0:
        result = "All sides"
        idx = np.where(x == True)[0][0]
        burn_results.loc[count, :]  = [int(meadow.ID), burn_shapefile.loc[idx, "YEAR_"], result]
        count += 1
    # if meadow only intersects burn geometry/geometries, at least one side burned
    else:
        y = burn_shapefile.intersects(area_burned)
        if sum(y) > 0:
            idx = list(np.where(y == True)[0])
            result = "At least 1 side"
            for idy in idx:
                burn_results.loc[count, :]  = [int(meadow.ID), int(burn_shapefile.loc[idy, "YEAR_"]), result]
                count += 1
    if meadowIdx % 500 == 0: print(meadowIdx, end=' ')

# write dataframe to csv file and filter shapefiles by burn csv to write to new shp
burn_results.to_csv("csv/Burned_meadows.csv", index=False)
merged_data = burn_results.merge(shapefile, on="ID", how="inner")
burn_shp = gpd.GeoDataFrame(merged_data, geometry="geometry", crs=shapefile.crs)
burn_shp.to_file("files/Burned_meadows.shp")
