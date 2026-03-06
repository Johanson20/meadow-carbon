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
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)
warnings.filterwarnings("ignore")


def computeStats(values):
    # function to calculate the slope, mean and sample standard deviation of time series values within a range of years
    stats = [[np.nan for _ in range(3)] for __ in range(len(values))]
    ans = []
    for val in values:
        if len(values[val]) > 1:
            ans.append([np.polyfit(val, values[val], 1)[0], np.mean(values[val]), np.std(values[val], ddof=1)])
    stats = ans
    return stats


# load shapefile of burn summaries and csv file of burned meadows (entirely within) already generated
epsg_crs = "EPSG:4326"
meadows = gpd.read_file("files/meadowsFiresCombos_1984to2024_20251208.shp").to_crs(epsg_crs)
meadows = meadows[meadows.rltnshp == "entirely_within"]
meadows.drop_duplicates(subset=['UniquID', 'YEAR_'], inplace=True)
meadows.sort_values(by="YEAR_", inplace=True)
meadows.reset_index(drop=True, inplace=True)
data = pd.read_csv("csv/burned_meadows_1984_2025.csv")
data = data[data.Description == "entirely_within"]
data.drop_duplicates(subset=['UniqueID', 'FireYear'], inplace=True)
data.reset_index(drop=True, inplace=True)
# select relevant columns and unique IDs for timeseries
uniqueIds = list(set(data.UniqueID))
mycols = ['X', 'Y', 'ANPP', 'BNPP', 'Rh', 'NEP']
cols = mycols[2:]
years = list(range(1984, 2025))

# create empty dataframe for storing statistics
all_cols = ['UniqueID', 'CalFireID', 'Relationship', 'FireYear']
for col in cols:
    all_cols.extend([f'{col}_Slope_Bfr_Fire', f'{col}_Mean_Bfr_Fire', f'{col}_Std_Bfr_Fire', f'{col}_Slope_Aft_Fire', f'{col}_Mean_Aft_Fire', f'{col}_Std_Aft_Fire'])
meadow_data = pd.DataFrame(columns=all_cols)


def generateFireTimeSeries(uniqueId):
    # for each unique ID, extract the feature and geometry
    feature = data[data.UniqueID == uniqueId]
    meadow = meadows[meadows.UniquID == uniqueId].iloc[0]
    meadow_geom = gpd.GeoDataFrame([meadow], geometry='geometry', crs=epsg_crs)
    minx, miny, maxx, maxy = meadow_geom.total_bounds
    ans = []    # final result/row(s) to be returned
    vals = {}   # initialize dictionary with values for each relevant column
    for col in cols:
        vals[col] = {}
    
    for year in years:
        df = []
        # load csv pixel level data in chunks of 50,000 rows and filter, as file is too large to load into memory at once
        for chunk in pd.read_csv(f"files/results/{year}_Meadows.csv", usecols = mycols, chunksize=50000):
            filtered = chunk[(chunk["X"] >= minx) & (chunk["X"] <= maxx) & (chunk["Y"] >= miny) & (chunk["Y"] <= maxy)]
            df.append(filtered)
        df = pd.concat(df, ignore_index=True)
        if df.empty:    # skip if geometry had no values from csv file
            continue
        pixels_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=epsg_crs)
        # spatially join feature's geometry with carbon model values from csv
        joined = gpd.sjoin(pixels_gdf, meadow_geom, how='inner', predicate='intersects')
        stats = joined.groupby('index_right')[cols].agg('mean')
        if stats.empty: continue
        for col in cols:
            vals[col][year] = stats[col].values[0]
        # if year%10 == 0: print(year, end=', ')    # print progress every decade
    del chunk, filtered, df, pixels_gdf, joined, meadow_geom
    
    if not vals:    # skip processing if no valid values for geometry was found
        print(f"{uniqueId} returned no valid values!")
        return ans
    
    try:
        # prepare results for each unique fire year
        for idx in range(len(set(feature.FireYear))):
            fire = feature.iloc[idx]
            ans.append([meadow.UniquID, meadow.CalFrID, fire.Description, fire.FireYear])
        # write plots to a pdf
        with PdfPages(f"files/fire/{uniqueId}_fire_timeseries.pdf") as pdf:
            for col in cols: 
                # extract fire dates for feature geometry for determining date range of timeseries analysis
                target_dates = iter([yr for yr in feature.FireYear])
                startyear = 1984
                target_date = next(target_dates, None)
                timerange, fireyears = {}, []
                while target_date:  # iterate through each unique fire date of the meadow and get stats between dates
                    year = int(target_date)
                    fireyears.append(target_date)
                    yrs = [yr for yr in range(startyear, year+1) if yr in vals[col]]
                    yCol = [vals[col][yr] for yr in yrs]
                    timerange[tuple(yrs)] = yCol
                    startyear = year+1
                    target_date = next(target_dates, None)
                timerange[tuple([yr for yr in range(year+1, 2025) if yr in vals[col]])] = [vals[col][yr] for yr in range(startyear, 2025)]
                # compute stats for each time range
                results = computeStats(timerange)
            
                # plot time series graph
                plt.figure(figsize=(8, 6))
                plt.plot([yr for yr in years if yr in vals[col]], vals[col].values(), marker="o")
                # Plot vertical line for each fire year
                for yr in fireyears:
                    plt.axvline(yr, linestyle="--")
                # Legend with stats
                legend_lines = []
                for firedate, result in zip(fireyears, results):
                    slope, mean, std = result
                    legend_lines.append(f"Before {firedate}: slope = {slope:.2f}, mean = {mean:.2f}, stdev = {std:.2f}")
                slope, mean, std = results[-1]
                legend_lines.append(f"After {firedate}: slope = {slope:.2f}, mean = {mean:.2f}, stdev = {std:.2f}")
                legend_text = "\n".join(legend_lines)
                plt.title(legend_text, fontsize=10)
                plt.xlabel("Year", fontweight='bold')
                plt.ylabel(f"{col}", fontweight='bold')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            # store records for dataframe
            try:
                for idx in range(len(set(feature.FireYear))):
                    fire = feature.iloc[idx]
                    ans[idx].extend((results[idx] + results[idx+1]))
            except:     # error expected to be thrown if fire occurs in 2024 (no data after)
                pass
        print(uniqueId, end=', ')
    except:
        print(f"{uniqueId} threw an error!")
    
    return ans


# generateFireTimeSeries("SNM6702")
with Parallel(n_jobs=18, prefer="threads") as parallel:
    results = parallel(delayed(generateFireTimeSeries)(uniqueId) for uniqueId in uniqueIds)

# write finals rows from results into csv file
for idx in range(len(results)):
    result = results[idx]
    if result and len(result) > 1:
        for res in result:
            if len(res) > 4: meadow_data.iloc[len(meadow_data), :] = res
    else:
        if result and len(result) > 4: meadow_data.iloc[len(meadow_data), :] = result
meadow_data.to_csv("csv/burned_meadows_statistics.csv", index=False)
