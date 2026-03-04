# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:24:10 2026

@author: jonyegbula
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)
warnings.filterwarnings("ignore")


def computeStats(values):
    # function to calculate the slope, mean and sample standard deviation of time series values within a range of years
    stats = [[np.nan]*3]*len(values)
    ans = []
    for val in values:
        if len(values[val]) > 1:
            ans.append([np.polyfit(val, values[val], 1)[0], np.mean(values[val]), np.std(values[val], ddof=1)])
    stats = ans
    return stats


# load shapefile of burn summaries and csv file of burned meadows (touching or within) already generated
epsg_crs = "EPSG:4326"
meadows = gpd.read_file("files/meadows_dateBeforeFire_fires2012to2023_20251210.shp").to_crs(epsg_crs)
data = pd.read_csv("files/fire_meadows_data.csv")
data = data[data.Description != "within_100m"]
# select relevant columns and unique IDs for timeseries
uniqueIds = list(set(data.UniqueID))
mycols = ['X', 'Y', 'NEP']
col = "NEP"
years = list(range(1984, 2025))
dates = pd.to_datetime([f"{y}-10-01" for y in years])


def generateFireTimeSeries(uniqueId):
    # for each unique ID, extract the feature and geometry
    feature = meadows[meadows.UniqueID == uniqueId]
    meadow_geom = gpd.GeoDataFrame(geometry=feature.geometry, crs=epsg_crs)
    minx, miny, maxx, maxy = meadow_geom.total_bounds
    vals = {}
    
    for year in years:
        # load csv pixel level data in chunks of 50,000 rows and filter, as file is too large to load into memory at once
        for df in pd.read_csv(f"files/results/{year}_Meadows.csv", usecols = mycols, chunksize=50000):
            df = df[(df["X"] >= minx) & (df["X"] <= maxx) & (df["Y"] >= miny) & (df["Y"] <= maxy)]
            if df.empty:    # skip if geometry had no values from csv file
                continue
            pixels_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs="EPSG:4326")
        # spatially join feature's geometry with carbon model values from csv
        joined = gpd.sjoin(pixels_gdf, meadow_geom, how='inner', predicate='within')
        stats = joined.groupby('index_right')[col].agg('mean')
        if stats.empty: continue
        vals[year] = stats.values[0]
        if year%10 == 0: print(year, end=', ')    # print progress every decade

    if not vals or len(vals) < 10:    # skip processing if insufficient valid values for geometry was found
        print(f"{uniqueId} returned no valid {col} values!")
        return
    
    # extract fire dates for feature geometry for determining date range of timeseries analysis
    target_dates = iter([date.strftime("%Y-%m-%d") for date in feature.DtFire])
    startyear = 1984
    target_date = next(target_dates, None)
    timerange, fireyears = {}, []
    while target_date:  # iterate through each unique fire date of the meadow and get stats between dates
        year = int(target_date[:4])
        fireyears.append(target_date)
        yrs = range(startyear, year+1)
        yCol = [vals[yr] for yr in yrs]
        timerange[yrs] = yCol
        startyear = year+1
        target_date = next(target_dates, None)
    timerange[range(year+1, 2025)] = [vals[yr] for yr in range(startyear, 2025)]
    # compute stats for each time range
    results = computeStats(timerange)
    
    # plot time series graph
    plt.figure(figsize=(8, 5))
    plt.plot(dates, vals.values(), marker="o")
    # Plot vertical line for each fire year
    for yr in fireyears:
        plt.axvline(pd.to_datetime(yr), linestyle="--")
    # Legend with stats
    legend_lines = []
    for firedate, result in zip(fireyears, results):
        slope, mean, std = result
        legend_lines.append(f"Before {firedate}: slope = {slope:.2f}, mean = {mean:.2f}, stdev = {std:.2f}")
    slope, mean, std = results[-1]
    legend_lines.append(f"After {firedate}: slope = {slope:.2f}, mean = {mean:.2f}, stdev = {std:.2f}")
    legend_text = "\n".join(legend_lines)
    
    plt.legend([legend_text])
    plt.title(f"{col} Time Series for {uniqueId}")
    plt.xlabel("Year")
    plt.ylabel(f"{col}")
    plt.tight_layout()
    plt.show()


# generateFireTimeSeries("SNM415")
for uniqueId in uniqueIds:
    generateFireTimeSeries(uniqueId)
