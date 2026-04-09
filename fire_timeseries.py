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
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)
warnings.filterwarnings("ignore")


# function to calculate the slope, mean and sample standard deviation of time series values within a range of years
def computeStats(values):
    output = []
    for val in values:
        if len(values[val]) > 0:
            output.append([np.polyfit(val, values[val], 1)[0], np.mean(values[val]), np.std(values[val], ddof=1)])
    if len(output) < 2 or len(output) < len(values):
        output.extend([[np.nan]*3])
    return output


# function to check if every fire date has a non-fire 5 year range about it
def check5YearRange(fireIncident):
    # drop duplicated rows where multiple fires occur in same year
    fireIncident.drop_duplicates(subset='FireYear', inplace=True)
    allyear = set()
    # check if fire year is at least 5 years past 1984 or 5 years before 2024
    for year in fireIncident.FireYear:
        if ((year - 1984) < 5) or ((2024 - year) < 5):
            fireIncident = fireIncident[fireIncident['FireYear'] != year]
    # check if other fire years (if available) have 5 year ranges before and after
    if not fireIncident.empty and len(fireIncident) > 2:
        oldYear, j = fireIncident.FireYear.iloc[0], 2
        for year in fireIncident[1:-1].FireYear:
            nextYear = fireIncident.FireYear.iloc[j]
            if ((year - oldYear) < 5) or ((nextYear - year) < 5):
                fireIncident = fireIncident[fireIncident['FireYear'] != year]
            oldYear, j = year, j + 1
    for year in fireIncident.FireYear: 
        for yr in range(year-5, year+6): allyear.add(yr)
    return fireIncident, sorted(list(allyear))


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

allyears = {}
for yr in years:
    allyears[yr] = pd.read_csv(f"files/results/{yr}_Meadows.csv")
    print(yr, end=' ')


def generateFireTimeSeries(uniqueId):
    # for each unique ID, extract the feature and geometry
    fireIncident = data[data.UniqueID == uniqueId]
    meadow = meadows[meadows.UniquID == uniqueId].iloc[0]
    meadow_geom = gpd.GeoDataFrame([meadow], geometry='geometry', crs=epsg_crs)
    minx, miny, maxx, maxy = meadow_geom.total_bounds
    ans = []    # final result/row(s) to be returned
    vals = {}   # initialize dictionary with values for each relevant column
    for col in cols:
        vals[col] = {}
    
    for year in years:
        df = allyears[year]
        df = df[(df["X"] >= minx) & (df["X"] <= maxx) & (df["Y"] >= miny) & (df["Y"] <= maxy)]
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
    del df, pixels_gdf, joined, meadow_geom
    
    if not vals:    # skip processing if no valid values for geometry was found
        print(f"{uniqueId} returned no valid values!")
        return ans
    
    try:
        # prepare results for each unique fire year
        for idx in range(len(set(fireIncident.FireYear))):
            fire = fireIncident.iloc[idx]
            ans.append([meadow.UniquID, meadow.CalFrID, fire.Description, fire.FireYear])
        # write plots to a pdf
        with PdfPages(f"files/fire/{uniqueId}_fire_timeseries.pdf") as pdf:
            for col in cols: 
                # extract fire dates for feature geometry for determining date range of timeseries analysis
                target_dates = iter([yr for yr in fireIncident.FireYear])
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
                yrs = [yr for yr in range(startyear, 2025) if yr in vals[col]]
                timerange[tuple(yrs)] = [vals[col][yr] for yr in yrs]
                # compute stats for each time range
                outcome = computeStats(timerange)
            
                # plot time series graph
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.plot([yr for yr in years if yr in vals[col]], vals[col].values(), marker="o")
                # Plot vertical line for each fire year
                for yr in fireyears:
                    ax.axvline(yr, linestyle="--")
                # Legend with stats
                legend_lines = []
                for firedate, result in zip(fireyears, outcome):
                    slope, mean, std = result
                    legend_lines.append(f"Before {firedate}: slope = {slope:.2f}, mean = {mean:.2f}, stdev = {std:.2f}")
                slope, mean, std = outcome[-1]
                legend_lines.append(f"After {firedate}: slope = {slope:.2f}, mean = {mean:.2f}, stdev = {std:.2f}")
                legend_text = "\n".join(legend_lines)
                ax.set_title(legend_text, fontsize=10)
                ax.set_xlabel("Year", fontweight='bold')
                ax.set_ylabel(f"{col}", fontweight='bold')
                pdf.savefig(fig)
                plt.close(fig)
                # store records for dataframe
                try:
                    for idx in range(len(set(fireIncident.FireYear))):
                        fire = fireIncident.iloc[idx]
                        ans[idx].extend((outcome[idx] + outcome[idx+1]))
                except:     # error expected to be thrown if fire occurs in 2024 (no data after)
                    print(f"Couldn't store records for {uniqueId}!")
    except:
        print(f"{uniqueId} error!")
    
    return ans


def generate_5_Year_TimeSeries(uniqueId):
    ans = []    # final result/row(s) to be returned
    # for each unique ID, extract the feature and geometry
    fireIncident = data[data.UniqueID == uniqueId]
    try:
        fireIncident, myyears = check5YearRange(fireIncident)
    except:
        print(f"{uniqueId} returned error during 5 year range check!")
        return ans
    if fireIncident.empty or not myyears:    # skip processing if no 5 year ranges were found for fire dates
        return ans
    
    try:
        meadow = meadows[meadows.UniquID == uniqueId].iloc[0]
        meadow_geom = gpd.GeoDataFrame([meadow], geometry='geometry', crs=epsg_crs)
        minx, miny, maxx, maxy = meadow_geom.total_bounds
        vals = {}   # initialize dictionary with values for each relevant column
        for col in cols:
            vals[col] = {}
        
        for year in myyears:
            df = allyears[year]
            df = df[(df["X"] >= minx) & (df["X"] <= maxx) & (df["Y"] >= miny) & (df["Y"] <= maxy)]
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
        del df, pixels_gdf, joined, meadow_geom
    except:
        print(f"Failed process for {uniqueId}")
        return ans
    
    if not vals:    # skip processing if no valid values for geometry was found
        print(f"{uniqueId} returned no valid values!")
        return ans
    
    try:
        # prepare results for each unique fire year
        for idx in range(len(set(fireIncident.FireYear))):
            fire = fireIncident.iloc[idx]
            ans.append([meadow.UniquID, meadow.CalFrID, fire.Description, fire.FireYear])
        # write plots to a pdf
        with PdfPages(f"files/fire/{uniqueId}_5year_timeseries.pdf") as pdf:
            for col in cols: 
                # extract fire dates for feature geometry for determining date range of timeseries analysis
                target_dates = iter([yr for yr in fireIncident.FireYear])
                target_date = next(target_dates, None)
                timerange, fireyears = {}, []
                while target_date:  # iterate through each unique fire date of the meadow and get stats between dates
                    year = int(target_date)
                    fireyears.append(target_date)
                    # extract valid values for 5 years before and after each fire date
                    for myrange in [range(year-5, year+1), range(year, year+6)]:
                        yrs = [yr for yr in myrange if yr in vals[col]]
                        yCol = [vals[col][yr] for yr in yrs]
                        timerange[tuple(yrs)] = yCol
                    target_date = next(target_dates, None)
                # compute stats for each time range
                outcome = computeStats(timerange)
            
                # plot time series graph
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.plot([yr for yr in myyears if yr in vals[col]], vals[col].values(), marker="o")
                # Plot vertical line for each fire year
                markYears = []
                for yr in fireyears:
                    markYears += [yr-5, yr, yr+5]
                for yr in markYears:
                    if yr in fireyears:
                        ax.axvline(yr, linestyle="--")
                    else:
                        ax.axvline(yr, linestyle="--", color='orange')
                # Legend with stats
                legend_lines = []
                for i in range(len(outcome)):
                    slope, mean, std = outcome[i]
                    firedate = fireyears[i//2]
                    if i%2== 0:
                        legend_lines.append(f"Before {firedate}: slope = {slope:.2f}, mean = {mean:.2f}, stdev = {std:.2f}")
                    else:
                        legend_lines.append(f"After {firedate}: slope = {slope:.2f}, mean = {mean:.2f}, stdev = {std:.2f}")
                legend_text = "\n".join(legend_lines)
                ax.set_title(legend_text, fontsize=10)
                ax.set_xlabel("Year", fontweight='bold')
                ax.set_ylabel(f"{col}", fontweight='bold')
                pdf.savefig(fig)
                plt.close(fig)
                # store records for dataframe
                try:
                    for idx in range(0, len(set(fireIncident.FireYear)), 2):
                        fire = fireIncident.iloc[idx//2]
                        ans[idx].extend((outcome[idx] + outcome[idx+1]))
                except:     # error expected to be thrown if fire occurs in 2024 (no data after)
                    print(f"Couldn't store records for {uniqueId}!")
    except:
        print(f"{uniqueId} error!")
    
    return ans


# generateFireTimeSeries("SNM6702")
# generate_5_Year_TimeSeries("SNM11781")

start = datetime.now()
# with Parallel(n_jobs=18, prefer="threads") as parallel:
    # results = parallel(delayed(generateFireTimeSeries)(uniqueId) for uniqueId in uniqueIds)
with Parallel(n_jobs=18, prefer="threads") as parallel:
    results = parallel(delayed(generate_5_Year_TimeSeries)(uniqueId) for uniqueId in uniqueIds)
print(f"Processing of timeseries completed in {datetime.now() - start}")

# write finals rows from results into csv file
for idx in range(len(results)):
    result = results[idx]
    if result and len(result) > 1:
        for res in result:
            if len(res) > 4: meadow_data.loc[len(meadow_data)] = res
    else:
        if result and len(result[0]) > 4: meadow_data.loc[len(meadow_data)] = result[0]
# meadow_data.to_csv("csv/burned_meadows_statistics.csv", index=False)
meadow_data.to_csv("csv/burned_meadows_5_year_stats.csv", index=False)
