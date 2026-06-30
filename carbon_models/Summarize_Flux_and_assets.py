# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:48:37 2026

@author: jonyegbula
"""

# This is for predicting soil and percent carbon
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
import pickle
from geocube.api.core import make_geocube

warnings.filterwarnings("ignore")

epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/AllPossibleMeadows_2025-10-22.shp").to_crs(epsg_crs)
# identify each meadow as UTM Zone 10 or 11
utm_zone10 = gpd.read_file("files/CA_UTM10.shp").to_crs(epsg_crs)
allIds = list(gpd.overlay(shapefile, utm_zone10, how="intersection").ID)

def makeFluxPredictions(year, makeStatic=False):
    '''
    This regenerates NEP and predictions and standard errors (mergeToSingleFile function) into grouped CSVs 
    '''
    
    mycols = ['ID', 'Jepson_Region', 'X', 'Y', 'ANPP', 'BNPP', 'Rh', 'NEP', '1SD_ANPP', '1SD_BNPP', '1SD_NEP', '1SD_Rh', 'Annual_Precipitation', 'AET', 'Active_growth_days', 'Minimum_temperature', 'Maximum_temperature', 'SRad', 'SWE', 'Wet_days', 'NDVI_June', 'NDWI_June', 'EVI_June', 'SAVI_June', 'BSI_June', 'NDPI_June', 'NDGI_June', 'NDVI_Sept', 'NDWI_Sept', 'EVI_Sept', 'SAVI_Sept', 'BSI_Sept', 'NDPI_Sept', 'NDGI_Sept', 'dBlue', 'dGreen', 'dRed', 'dNIR', 'dSWIR_1', 'dSWIR_2', 'dNDVI', 'dNDWI', 'dEVI', 'dSAVI', 'dBSI', 'dNDPI', 'dNDGI', 'Elevation', 'Slope', 'Shallow_Clay', 'Deep_Clay', 'Shallow_Sand', 'Deep_Sand', 'Shallow_Hydra_Conduc', 'Deep_Hydra_Conduc', 'Organic_Matter']
    
    # load all models and standard errors
    with open('files/carbon_models.pckl', 'rb') as f:
        ghg_model, agb_model, bgb_model = pickle.load(f)
    with  open('files/carbon_sd_models.pckl', 'rb') as f:
        ghg_84_model, agb_84_model, bgb_84_model = pickle.load(f)
    _, agb_col, bgb_col = list(ghg_model.feature_names_in_), list(agb_model.feature_names_in_), list(bgb_model.feature_names_in_)
    _, agb_sd_col, bgb_sd_col = list(ghg_84_model.feature_names_in_), list(agb_84_model.feature_names_in_), list(bgb_84_model.feature_names_in_)
    
    # differentiate the different kinds of output
    flux_col = mycols[:12]
    var_col = mycols[:4] + mycols[12:-9]
    static_col = mycols[:4] + mycols[-9:]
    flux_outfile = f"files/results/{year}_Meadow_flux.csv"
    var_outfile = f"files/results/{year}_Meadows.csv"
    static_outfile = "files/results/Meadow_static_variables.csv"
    all_flux_data = pd.DataFrame(columns=flux_col)
    all_var_data = pd.DataFrame(columns=var_col)
    all_static_data = pd.DataFrame(columns=static_col)
    
    # create summary statistics dataframe for each variable
    statCol = ['ID', 'PixelCount']
    for col in mycols[2:20] + ['NEP_Cap_1000']:
        if col in mycols[2:4]:
            statCol.append(col+"_mean")
        else:
            statCol.extend([col+"_mean", col+"_std"])
    stats_df = pd.DataFrame(columns=(statCol + ["NEP_sum", "Jepson_Region"]))
    all_files = [f for f in glob.glob(f"files/{year}/*.csv")]
    
    # iterate through each meadow's CSV to extract data
    for idx in range(len(all_files)):
        try:
            file = all_files[idx]
            zone = 32610 if int(file.split("_")[2][:-4]) in allIds else 32611
            meadowId = int(file.split("_")[2][:-4])
            jepson = shapefile[shapefile.ID == meadowId].EcoRegion.values[0]
            df = pd.read_csv(file)
            
            # Predict AGB/BGB and set negative values to zero, then convert to NEP
            rh_draws = np.random.normal(df['Rh'].to_frame(), df['1SD_Rh'].to_frame(), size=(len(df['Rh']), 100))
            df['HerbBio.g.m2'] = agb_model.predict(df.loc[:, agb_col])
            df['Roots.kg.m2'] = bgb_model.predict(df.loc[:, bgb_col])
            sd_agb = agb_84_model.predict(df.loc[:, agb_sd_col])
            df['1SD_ANPP'] = abs(sd_agb - df['HerbBio.g.m2'])
            sd_bgb = bgb_84_model.predict(df.loc[:, bgb_sd_col])
            bgb_sd = abs(sd_bgb - df['Roots.kg.m2'])
            df.loc[df['HerbBio.g.m2'] < 0, 'HerbBio.g.m2'] = 0
            df.loc[df['Roots.kg.m2'] < 0, 'Roots.kg.m2'] = 0
            df['Root_Turnover'] = (df['Roots.kg.m2']*0.49 - ((df['Roots.kg.m2']*0.49)*np.exp(-0.53)))*0.368*1000
            df['Root_Exudates'] = df['Roots.kg.m2']*1000*df['Active_growth_days']*12*1.04e-4
            df['BNPP'] = df['Root_Turnover'] + df['Root_Exudates']
            df['ANPP'] = df['HerbBio.g.m2']*0.433
            df['1SD_BNPP'] = (bgb_sd*0.49 - ((bgb_sd*0.49)*np.exp(-0.53)))*368 + bgb_sd*df['Active_growth_days']*12*0.104
            anpp_draws = np.random.normal(df['ANPP'].to_frame(), df['1SD_ANPP'].to_frame(), size=(len(df['ANPP']), 100))
            bnpp_draws = np.random.normal(df['BNPP'].to_frame(), df['1SD_BNPP'].to_frame(), size=(len(df['BNPP']), 100))
            df['NEP'] = df['ANPP'] + df['BNPP'] - df['Rh']
            df['1SD_NEP'] = pd.Series(np.std((anpp_draws + bnpp_draws - rh_draws), axis=1))
            # predict soil/percent C and set negative values to zero
            df['NEP_Cap_1000'] = [val if val <= 1000 else 1000 for val in df['NEP']]
            
            val = [meadowId, df.shape[0]]
            for col in mycols[2:20] + ['NEP_Cap_1000']:
                stats = df[col].describe().values
                val.extend(list(stats[1:4]) + [stats[-1]])
                if col == 'NEP': nep_sum = 900*stats[0]*stats[1]
            stats_df.loc[len(stats_df)] = val + [nep_sum, jepson]
            
            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df['X'], df['Y']), crs=zone).to_crs(4326)
            df['X'], df['Y'] = [p.x for p in gdf.geometry], [p.y for p in gdf.geometry]
            df[['ID', 'Jepson_Region']] = meadowId, jepson
            all_flux_data = pd.concat([all_flux_data, df[flux_col]], ignore_index=True)
            all_var_data = pd.concat([all_var_data, df[var_col]], ignore_index=True)
            if makeStatic:
                all_static_data = pd.concat([all_static_data, df[static_col]], ignore_index=True)
            if idx%1000 == 0: print(idx, end=" ")
        except: # fix wrongly appended NDVI column issues (x and y suffixes)
            if "NDVI_June_y" in df.columns:
                df.columns = list(df.columns)[:-2] + ["NDVI_June", "NDVI_Sept"]
                idx -= 1
            continue    
    
    del sd_agb, sd_bgb, bgb_sd, anpp_draws, bnpp_draws, rh_draws
    # write all files to CSVs
    stats_df.to_csv(var_outfile.split(".")[0] + "_stats.csv", index=False)
    all_flux_data = all_flux_data.dropna().drop_duplicates().reset_index(drop=True)
    all_var_data = all_var_data.dropna().drop_duplicates().reset_index(drop=True)
    all_flux_data.to_csv(flux_outfile, index=False)
    all_var_data.to_csv(var_outfile, index=False)
    if makeStatic:
        all_static_data = all_static_data.dropna().drop_duplicates().reset_index(drop=True)
        all_static_data.to_csv(static_outfile, index=False)
    del df, gdf, stats_df, all_flux_data, all_var_data, all_static_data
    
    return year

'''
for year in range(1985, 2025):
    makeFluxPredictions(year, True) if year == 2024 else remakeFinalPredictions(year)
'''


def splitCSVToGeotiffs(inputdir, attributes=None, zone=4326, res=30):
    ''' This function splits a csv file  into separate geotiffs with a column represented as its own geotiff as a band '''
    df = pd.read_csv(inputdir)
    utm_lons, utm_lats = df['X'], df['Y']
    if not attributes:
        attributes = list(df.columns[2:])   # assumption is first 2 columns of csv are spatial coordinates
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats), crs=zone).to_crs(3310)
    mycrs = "EPSG:" + str(zone)
    
    for attribute in attributes:
        gdf[attribute] = df[attribute]
        out_grd = make_geocube(vector_data=gdf, measurements=[attribute], resolution=(-res, res))
        out_grd = out_grd.rio.reproject(mycrs)
        out_grd = out_grd.astype("float32").chunk({"x": 2048, "y": 2048})
        out_grd.rio.to_raster((inputdir[:11] + attribute + ".tif"), tiled=True, compress="LZW", dtype="float32")
        gdf.drop(attribute, axis=1, inplace=True)
        df.drop(attribute, axis=1, inplace=True)
        print(attribute, "done!")

# splitCSVToGeotiffs("files/results/2021_Meadow_flux.csv", ['NEP', 'ANPP', 'BNPP', 'Rh'])
