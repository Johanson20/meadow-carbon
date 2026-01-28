# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:35:44 2025

@author: Johanson C. Onyegbula
"""

import os
import ee
import pickle
import warnings
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"      # adjust directory
os.chdir(mydir)

# read csv file and convert dates from strings to datetime
filename = "csv/Soil Carbon_RS Model.csv"
data = pd.read_csv(filename)
data.head()
data.drop_duplicates(inplace=True)  # remove duplicate rows

data.loc[:, ['Longitude', 'Latitude', 'SampleDate']].isna().sum()   # should be 0 for all columns
nullIds =  data[data[['Longitude', 'Latitude', 'SampleDate']].isna().any(axis=1)].index    # rows with null coordinates/dates
data.drop(nullIds, inplace = True)
# fix coordinate values that are often mistyped
data.columns
data.loc[:, ['Longitude', 'Latitude']].describe()
idx = data[data['Longitude'] > -116].index
data.loc[idx, 'Longitude'] -= 100
data.to_csv(filename, index=False)
data.reset_index(drop=True, inplace=True)
# adjust datetime format
data['SampleDate'] = [pd.to_datetime(x).strftime("%Y-%m-%d") for x in data['SampleDate']]
data.head()

# Authenticate and Initialize the Earth Engine API
#ee.Authenticate()
ee.Initialize()
warnings.filterwarnings("ignore")


def calculateIndices(image):
    # normalize raw reflectance values
    scaled_bands = image.select(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2']).multiply(2.75e-05).add(-0.2)
    image = image.addBands(scaled_bands, overwrite=True)
    
    # add indices
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndmi = image.normalizedDifference(['NIR', 'SWIR_1']).rename('NDMI')
    ndwi = image.normalizedDifference(['Green', 'NIR']).rename('NDWI')
    evi = image.expression("2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'BLUE': image.select('Blue')}).rename('EVI')
    savi = image.expression("1.5 * ((NIR - RED) / (NIR + RED + 0.5))", {'NIR': image.select('NIR'), 'RED': image.select('Red')}).rename('SAVI')
    bsi = image.expression("((RED + SWIR_1) - (NIR + BLUE)) / (RED + SWIR_1 + NIR + BLUE)", {'RED': image.select('Red'), 'SWIR_1': image.select('SWIR_1'), 'NIR': image.select('NIR'), 'BLUE': image.select('Blue')}).rename('BSI')
    ndpi = image.expression("(NIR - ((0.56 * RED) + (0.44 * SWIR_2))) / (NIR + ((0.56 * RED) + (0.44 * SWIR_2)))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'SWIR_2': image.select('SWIR_2')}).rename('NDPI')
    
    return image.addBands([ndvi, ndmi, ndwi, evi, savi, bsi, ndpi])


def maskCloud(image):
    # rename bands and mask out cloud based on bits in QA_pixel
    image = image.rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'QA'])
    qa = image.select('QA')
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = dilated_cloud.And(cirrus).And(cloud).And(cloudShadow)
    return image.updateMask(cloud_mask).select(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2'])


def getBandValues(image):
    # extract band values (with indices)
    image = calculateIndices(image)
    values = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30)
    date = image.date().format('YYYY-MM-dd')
    return ee.Feature(None, values).set('Date', date)


def gridmetValues(image):
    values = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30)
    date = image.date().format('YYYY-MM-dd')
    return ee.Feature(None, values).set('Date', date)


# get band values at peak EVI date, number of wet and snow days, as well as integrals over growing season
def extractActiveGrowth(landsat, target_date):
    # get band values and ensure it is not null
    year, month, day = target_date.split("-")
    prev_5_year = f"{str(int(year)-5)}-{month}-{day}"
    landsat_values = landsat.map(getBandValues).getInfo()['features']
    if not landsat_values:
        return [[None]*13, None]
    if x >= -120:   # latitudes between 120W and 114W refer to EPSG:32611"
        temp_val = gridmet_11.filterBounds(point).filterDate(prev_5_year, target_date)
    else:
        temp_val = gridmet.filterBounds(point).filterDate(prev_5_year, target_date)
    gridmet_values = temp_val.map(gridmetValues).getInfo()['features']
    L_values, G_values = [], []
    # convert band values and other properties to a dataframe for manipulation
    for feature in landsat_values:
        L_values.append(feature['properties'])
    for feature in gridmet_values:
        G_values.append(feature['properties'])
    df = pd.DataFrame(L_values)
    df_grid = pd.DataFrame(G_values)
    df['Date'] = pd.to_datetime(df['Date'])
    df_grid['Date'] = pd.to_datetime(df_grid['Date'])
    df.drop_duplicates(subset='Date', inplace=True)
    df.dropna(inplace=True)
    
    # Set the 'date' column as the index, re-index and linearly interpolate Landsat/Gridmet bands at daily frequency
    df_daily = df.set_index('Date')
    df_grid = df_grid.set_index('Date')
    date_range = pd.date_range(start=prev_5_year, end=target_date, freq='D')[:-1]
    df_daily = df_daily.reindex(date_range).interpolate(method='linear').ffill().bfill()
    df_daily.dropna(inplace=True)
    df_5_years = pd.concat([df_daily, df_grid], axis=1)
    df_5_years['NDSI'] = (df_5_years['Green'] - df_5_years['SWIR_1'])/(df_5_years['Green'] + df_5_years['SWIR_1'])
    
    # Active growing season is also when NDVI >= 0.2 without snow or water coverage, at above zero temperature
    df_non_snow = df_5_years[(df_5_years.NDSI <= 0.2)]
    df_non_snow['Year'] = df_non_snow.index.year
    df_growthdays = df_non_snow[(df_non_snow.NDVI > 0.2) & (df_non_snow.NDWI <= 0.5) & (df_non_snow.tmmn > 273.15)]
    df_growthdays['NDVI_Ratio'] = (df_growthdays.groupby('Year')['NDVI'].transform(lambda x: (x - x.min()) / (x.max() - x.min())))
    # filter days before 7/15 where NDVI ratio > 0.2, and days from 7/15 where it is greater than 0.6
    is_early = ((df_growthdays.index.month < 7) |  ((df_growthdays.index.month == 7) & (df_growthdays.index.day < 15)))
    is_late = ~is_early
    active_growth_days = (((is_early) & (df_growthdays['NDVI_Ratio'] > 0.2)) | ((is_late) & (df_growthdays['NDVI_Ratio'] > 0.6))).sum()/5
    # compute number of snow days (NDSI > 0.2) and number of wet days (NDWI > 0.5), and growing season integrals
    integrals = df_growthdays.sum()/5
    return [integrals[:13], round(active_growth_days)]


def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=30)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=30)


# reads Landsat data, flow accumulation, daymet, terraclimate and DEM data (for slope and elevation)
landsat9_collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat7_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat_collection = landsat9_collection.merge(landsat8_collection).merge(landsat7_collection).map(maskCloud)
landsat_June = landsat_collection.filterDate("1999-06-01", "2024-06-30").filter(ee.Filter.calendarRange(6, 6, 'month'))
landsat_Sept = landsat_collection.filterDate("1999-09-01", "2024-09-30").filter(ee.Filter.calendarRange(9, 9, 'month'))

# flow accumulation (463.83m resolution); terraclimate (4638.3m resolution); slope and elevation (10.2m resolution); 
dem = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32610", scale=30)
slopeDem = ee.Terrain.slope(dem)
terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(['aet', 'pr', 'swe']).map(resample10)
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select(['tmmn', 'tmmx', 'srad']).map(resample10)

dem_11 = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32611", scale=30)
slopeDem_11 = ee.Terrain.slope(dem_11)
terraclimate_11 = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(['aet', 'pr', 'swe']).map(resample11)
gridmet_11 = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select(['tmmn', 'tmmx', 'srad']).map(resample11)

# these polaris soil datasets have 30m spatial resolution (same as landsat above); lithology is 90m resolution
perc_clay = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample10)
hydra_cond = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample10)
perc_sand = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').select("b1").map(resample10)
lithology = ee.Image("CSP/ERGo/1_0/US/lithology").select("b1").resample("bilinear").reproject(crs="EPSG:32610", scale=30)

shallow_perc_clay = ee.ImageCollection(perc_clay.toList(3)).mean()
deep_perc_clay = ee.Image(perc_clay.toList(6).get(3))
shallow_hydra_cond = ee.ImageCollection(hydra_cond.toList(3)).mean()
deep_hydra_cond = ee.Image(hydra_cond.toList(6).get(3))
shallow_perc_sand = ee.ImageCollection(perc_sand.toList(3)).mean()
deep_perc_sand = ee.Image(perc_sand.toList(6).get(3))

perc_clay_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample11)
hydra_cond_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample11)
perc_sand_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').select("b1").map(resample11)
lithology_11 = ee.Image("CSP/ERGo/1_0/US/lithology").select("b1").resample("bilinear").reproject(crs="EPSG:32611", scale=30)

shallow_perc_clay_11 = ee.ImageCollection(perc_clay_11.toList(3)).mean()
deep_perc_clay_11 = ee.Image(perc_clay_11.toList(6).get(3))
shallow_hydra_cond_11 = ee.ImageCollection(hydra_cond_11.toList(3)).mean()
deep_hydra_cond_11 = ee.Image(hydra_cond_11.toList(6).get(3))
shallow_perc_sand_11 = ee.ImageCollection(perc_sand_11.toList(3)).mean()
deep_perc_sand_11 = ee.Image(perc_sand_11.toList(6).get(3))

data[['BSI_June', 'Blue_June', 'EVI_June', 'Green_June', 'NDMI_June', 'NDPI_June', 'NDVI_June', 'NDWI_June', 'NIR_June',
      'Red_June', 'SAVI_June', 'SWIR_1_June', 'SWIR_2_June', 'BSI_Sept', 'Blue_Sept', 'EVI_Sept', 'Green_Sept',
      'NDMI_Sept', 'NDPI_Sept', 'NDVI_Sept', 'NDWI_Sept', 'NIR_Sept', 'Red_Sept', 'SAVI_Sept', 'SWIR_1_Sept',  'SWIR_2_Sept', 'dBSI', 'dBlue', 'dEVI', 'dGreen', 'dNDMI', 'dNDPI', 'dNDVI', 'dNDWI', 'dIR', 'dRed', 'dSAVI', 'dSWIR_1', 'dSWIR_2', 'Evapotranspiration', 'Precipitation', 'SWE', 'SRad', 'Min_Temp', 'Max_Temp', 'Active_growth_days', 'Elevation','Slope','Shallow_Clay','Shallow_Sand', 'Shallow_Hydra_Conduc', 'Deep_Clay', 'Deep_Sand', 'Deep_Hydra_Conduc', 'Lithology']] = None

# populate bands by applying above functions for each pixel in dataframe
def bandsRun(idx):
    try:
        # extract coordinates and date from csv
        global x, point
        x, y = data.loc[idx, ['Longitude', 'Latitude']]
        point = ee.Geometry.Point(x, y)
        target_date = data.loc[idx, 'SampleDate']
        year, month, day = target_date.split("-")
        prev_5_year = f"{str(int(year)-5)}-{month}-{day}"
        if int(year) > 2024:    # 2025 data is unavailable
            data.drop(idx, inplace=True)
            print("Row", idx, "dropped!")
        
        # compute 5 year average of landsat bands/indices in June and September
        June_landsat = calculateIndices(landsat_June.filterBounds(point).filterDate(prev_5_year, target_date).mean())
        Sept_landsat = calculateIndices(landsat_Sept.filterBounds(point).filterDate(prev_5_year, target_date).mean())
        bands_June = June_landsat.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30).getInfo()
        bands_Sept = Sept_landsat.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30).getInfo()
        landsat = landsat_collection.filterBounds(point).filterDate(prev_5_year, target_date)
        integrals, growth_days = extractActiveGrowth(landsat, target_date)
        
        # compute values from daymetv4 (1km resolution) and terraclimate (resolution of both is 4,638.3m)
        if x >= -120:   # latitudes between 120W and 114W refer to EPSG:32611"
            terra_values = terraclimate_11.filterBounds(point).filterDate(prev_5_year, year+"-12-31").mean()
            grid_values = gridmet_11.filterBounds(point).filterDate(prev_5_year, year+"-12-31").mean()
            elev = dem_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
            slope_value = slopeDem_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
            shallow_clay = shallow_perc_clay_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            deep_clay = deep_perc_clay_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            shallow_hydra = shallow_hydra_cond_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            deep_hydra = deep_hydra_cond_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            shallow_sand = shallow_perc_sand_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            deep_sand = deep_perc_sand_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            lith = lithology_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        else:
            terra_values = terraclimate.filterBounds(point).filterDate(prev_5_year, year+"-12-31").mean()
            grid_values = gridmet.filterBounds(point).filterDate(prev_5_year, year+"-12-31").mean()
            elev = dem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
            slope_value = slopeDem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
            shallow_clay = shallow_perc_clay.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            deep_clay = deep_perc_clay.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            shallow_hydra = shallow_hydra_cond.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            deep_hydra = deep_hydra_cond.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            shallow_sand = shallow_perc_sand.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            deep_sand = deep_perc_sand.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            lith = lithology.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
            
        avg_terra_values = terra_values.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30).getInfo()
        avg_grid_values = grid_values.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30).getInfo()
        if idx%50 == 0: print(idx, end=' ')
        return list(bands_June.values()) + list(bands_Sept.values()) + list(integrals) + list(avg_terra_values.values()) + list(avg_grid_values.values()) + [growth_days, elev, slope_value, shallow_clay, shallow_sand, shallow_hydra, deep_clay, deep_sand, deep_hydra, lith] 
    except:
        print(idx)
        return [None]*55


with Parallel(n_jobs=15, prefer="threads") as parallel:
    result = parallel(delayed(bandsRun)(meadowIdx) for meadowIdx in range(data.shape[0]))
cols = list(data.columns)
for idx in range(len(result)):
    data.loc[idx, cols[10:]] = result[idx]

data['Evapotranspiration'] *= 0.1
data['Min_Temp'] -= 273.15
data['Max_Temp'] -= 273.15
data.head()
# write updated dataframe to new csv file
data.to_csv(filename.split(".csv")[0] + "_Data.csv", index=False)


# ML training starts here
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt

# read csv containing random samples and model soil carbon
data = pd.read_csv('csv/Soil Carbon_RS Model_Data.csv')
data.head()
cols = data.columns
# cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)

# remove irrelevant columns for ML and determine X and Y variables
var_col = list(cols[10:-1])
y_field = 'SoilC.kg.m2'
# subdata excludes other measured values which can be largely missing (as we need to assess just one output at a time)
subdata = data.loc[:, ([y_field] + var_col)]
# check for missing/null values across columns and rows respectively (ideal results are typically 0)
sum(subdata.isnull().any(axis=0) == True)
sum(subdata[y_field].isnull())

# create correlation matrix
corr_mat = pd.DataFrame(index=np.arange(len(var_col)), columns=var_col)
for i in range(len(var_col)):
    vals = []
    col1 = var_col[i]
    for col2 in var_col:
        vals.append(np.corrcoef(data[col1], data[col2])[0][1])
    corr_mat.iloc[i, :] = vals
corr_mat.index = corr_mat.columns
corr_mat.to_csv("files/Soil_carbon_correlation.csv")

# if NAs where found (results above are not 0) in one of them (e.g. Y)
nullIds = list(np.where(subdata[y_field].isnull())[0])    # null IDs
data.drop(nullIds, inplace = True)
data.dropna(subset=[y_field], inplace=True)
data.dropna(subset=var_col, inplace=True)
data.reset_index(drop=True, inplace=True)
# make scatter plots of relevant variables from raw dataframe
with PdfPages('files/SoilC_Scatter_plots.pdf') as pdf:
    col2 = cols[10:]
    for i in range(0, len(col2), 9):
        # Get up to 9 features for this page
        page_features = col2[i:i+9]
        # Create 3x3 subplots
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11, 8.5))
        axes = axes.flatten()  # Flatten to 1D array for easy indexing
        for j, feature in enumerate(page_features):
            ax = axes[j]
            sns.regplot(x=feature, y=y_field, data=data, line_kws={"color":"red"}, ax=ax)
            ax.set_title(f'{feature} vs {y_field}', fontsize=10)
        # Hide unused subplots if fewer than 9 features on last page
        for j in range(len(page_features), 9):
            fig.delaxes(axes[j])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# bin the dataset based on soil carbon values
data[y_field].describe()
data['SoilC'] = 0
for value in range(0, 30, 3):   # ranges from 1.6 to 28.65
    mask = (data[y_field] > value) & (data[y_field] <= (value+3))
    data.loc[mask, 'SoilC'] = value//3
data['SoilC'].describe()

# split data into training (80%) and test data (20%) by IDs, random state ensures reproducibility
train_df = data.groupby('SoilC', group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=10))
test_data = data[~data.index.isin(train_df.index)]
# upsample the training dataset so that all bins have same amount of rows
max_size = train_df['SoilC'].value_counts().max()
train_data = (train_df.groupby('SoilC', group_keys=False)
    .apply(lambda x: resample(x, replace=True, n_samples=max_size, random_state=10)).reset_index(drop=True))
train_data['SoilC'].value_counts()

X_train, y_train = train_data.loc[:, var_col], train_data[y_field]
X_test, y_test = test_data.loc[:, var_col], test_data[y_field]

bgb_soilc_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, n_estimators=100, subsample=0.6,
                validation_fraction=0.2, n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
bgb_soilc_model.fit(X_train, y_train)
# Make partial dependence plots
with PdfPages('files/SoilC_partial_dependence_plots.pdf') as pdf:
    for i in range(0, len(var_col), 9):
        page_features = var_col[i:i+9]
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11, 8.5))
        axes = axes.flatten()
        for j, feature in enumerate(page_features):
            ax = axes[j]
            PartialDependenceDisplay.from_estimator(bgb_soilc_model, data[var_col], [feature], random_state=10, ax=ax)
            ax.set_title(f'Partial Dependence of {feature}', fontsize=10)
        # Remove unused axes
        for j in range(len(page_features), 9):
            fig.delaxes(axes[j])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

len(bgb_soilc_model.estimators_)  # number of trees used in estimation
# print relevant stats
y_train_pred = bgb_soilc_model.predict(X_train)
y_test_pred = bgb_soilc_model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mape = mean_absolute_percentage_error(y_train_pred, y_train)*100
train_corr = np.corrcoef(y_train, y_train_pred)
val = (y_train_pred - y_train) / y_train
train_p_bias = np.mean(val[np.isfinite(val)]) * 100

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test_pred, y_test)*100
test_corr = np.corrcoef(y_test, y_test_pred)
val = (y_test_pred - y_test) / y_test
test_p_bias = np.mean(val[np.isfinite(val)]) * 100

print("\nTRAINING DATA:\nRoot Mean Squared Error (RMSE) = {}\nMean Absolute Error (MAE) = {}".format(train_rmse, train_mae))
print("\nMean Absolute Percentage Error (MAPE) Over Predictions = {} %\nCorrelation coefficient matrix (R) = {}".format(train_mape, train_corr[0][1]))
print("\nTEST DATA:\nRoot Mean Squared Error (RMSE) = {}\nMean Absolute Error (MAE) = {}".format(test_rmse, test_mae))
print("\nMean Absolute Percentage Error (MAPE) Over Predictions = {} %\nCorrelation coefficient (R) = {}".format(test_mape, test_corr[0][1]))
print("\nMean Training Percentage Bias = {} %\nMean Test Percentage Bias = {} %".format(train_p_bias, test_p_bias))

# plot Feature importance
feat_imp = bgb_soilc_model.feature_importances_
sorted_idx = np.argsort(feat_imp)
pos = np.arange(sorted_idx.shape[0]) + 0.5
# Make regression line over y_test and it's predictions
regressor = LinearRegression()
y_test = np.array(y_test).reshape(-1,1)
y_test_pred = np.array(y_test_pred).reshape(-1,1)
regressor.fit(y_test, y_test_pred)
y_pred = regressor.predict(y_test)

def plotFeatureImportance():
    plt.barh(pos, feat_imp[sorted_idx], align="center")
    plt.yticks(pos, np.array(bgb_soilc_model.feature_names_in_)[sorted_idx])
    plt.title("Feature Importance")

def plotY():
    plt.scatter(y_test, y_test_pred, color='g')
    plt.plot(y_test, y_pred, color='k', label='Regression line')
    plt.plot(y_test, y_test, linestyle='dotted', color='gray', label='1:1 line')
    plt.xlabel('Actual ' + y_field)
    plt.ylabel("Predicted " + y_field)
    plt.title(f"Test set (y_test); R = {np.round(test_corr[0][1], 4)}")
    # Make axes of equal extents
    axes_lim = np.ceil(max(max(y_test), max(y_test_pred))) + 2
    plt.xlim((0, axes_lim))
    plt.ylim((0, axes_lim))
    plt.legend()

plotFeatureImportance()
plotY()


# read csv containing random samples and model percent carbon
data = pd.read_csv('csv/Soil Carbon_RS Model_Data.csv')
data.head()
cols = data.columns
# cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)

# remove irrelevant columns for ML and determine X and Y variables
var_col = list(cols[10:-1])
y_field = 'percentC'
# subdata excludes other measured values which can be largely missing (as we need to assess just one output at a time)
subdata = data.loc[:, ([y_field] + var_col)]
# check for missing/null values across columns and rows respectively (confirm results below should be all 0)
sum(subdata.isnull().any(axis=0) == True)
sum(subdata[y_field].isnull())

# create correlation matrix
corr_mat = pd.DataFrame(index=np.arange(len(var_col)), columns=var_col)
for i in range(len(var_col)):
    vals = []
    col1 = var_col[i]
    for col2 in var_col:
        vals.append(np.corrcoef(data[col1], data[col2])[0][1])
    corr_mat.iloc[i, :] = vals
corr_mat.index = corr_mat.columns
corr_mat.to_csv("files/Percent_carbon_correlation.csv")

# if NAs where found (results above are not 0) in one of them (e.g. Y)
nullIds = list(np.where(subdata[y_field].isnull())[0])    # null IDs
data.drop(nullIds, inplace = True)
data.dropna(subset=[y_field], inplace=True)
data.dropna(subset=var_col, inplace=True)
data.reset_index(drop=True, inplace=True)
# make scatter plots of relevant variables from raw dataframe
with PdfPages('files/PercentC_Scatter_plots.pdf') as pdf:
    for i in range(0, len(var_col), 9):
        # Get up to 9 features for this page
        page_features = var_col[i:i+9]
        # Create 3x3 subplots
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11, 8.5))
        axes = axes.flatten()  # Flatten to 1D array for easy indexing
        for j, feature in enumerate(page_features):
            ax = axes[j]
            sns.regplot(x=feature, y=y_field, data=data, line_kws={"color":"red"}, ax=ax)
            ax.set_title(f'{feature} vs {y_field}', fontsize=10)
        # Hide unused subplots if fewer than 9 features on last page
        for j in range(len(page_features), 9):
            fig.delaxes(axes[j])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# split data into training (80%) and test data (20%) by IDs, random state ensures reproducibility
data[y_field].describe()
data['percentCarbon'] = 0
for value in range(0, 40, 4):   # ranges from 0.85 to 39.33
    mask = (data[y_field] > value) & (data[y_field] <= (value+4))
    data.loc[mask, 'percentCarbon'] = value//4
data['percentCarbon'].describe()

# split data into training (80%) and test data (20%) by IDs, random state ensures reproducibility
train_df = data.groupby('percentCarbon', group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=10))
test_data = data[~data.index.isin(train_df.index)]
# upsample the training dataset so that all bins have same amount of rows
max_size = train_df['percentCarbon'].value_counts().max()
train_data = (train_df.groupby('percentCarbon', group_keys=False)
    .apply(lambda x: resample(x, replace=True, n_samples=max_size, random_state=10)).reset_index(drop=True))
train_data['percentCarbon'].value_counts()

X_train, y_train = train_data.loc[:, var_col], train_data[y_field]
X_test, y_test = test_data.loc[:, var_col], test_data[y_field]

bgb_percentc_model = GradientBoostingRegressor(learning_rate=0.07, max_depth=9, n_estimators=200, subsample=0.9,
                validation_fraction=0.2, n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
bgb_percentc_model.fit(X_train, y_train)
# Make partial dependence plots
with PdfPages('files/PercentC_partial_dependence_plots.pdf') as pdf:
    for i in range(0, len(var_col), 9):
        page_features = var_col[i:i+9]
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11, 8.5))
        axes = axes.flatten()
        for j, feature in enumerate(page_features):
            ax = axes[j]
            PartialDependenceDisplay.from_estimator(bgb_percentc_model, data[var_col], [feature], random_state=10, ax=ax)
            ax.set_title(f'Partial Dependence of {feature}', fontsize=10)
        # Remove unused axes
        for j in range(len(page_features), 9):
            fig.delaxes(axes[j])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

len(bgb_percentc_model.estimators_)  # number of trees used in estimation
# print relevant stats
y_train_pred = bgb_percentc_model.predict(X_train)
y_test_pred = bgb_percentc_model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mape = mean_absolute_percentage_error(y_train_pred, y_train)*100
train_corr = np.corrcoef(y_train, y_train_pred)
val = (y_train_pred - y_train) / y_train
train_p_bias = np.mean(val[np.isfinite(val)]) * 100

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test_pred, y_test)*100
test_corr = np.corrcoef(y_test, y_test_pred)
val = (y_test_pred - y_test) / y_test
test_p_bias = np.mean(val[np.isfinite(val)]) * 100

print("\nTRAINING DATA:\nRoot Mean Squared Error (RMSE) = {:.4f}\nMean Absolute Error (MAE) = {:.4f}".format(train_rmse, train_mae))
print("\nMean Absolute Percentage Error (MAPE) Over Predictions = {:.4f} %\nCorrelation coefficient matrix (R) = {:.4f}".format(train_mape, train_corr[0][1]))
print("\nTEST DATA:\nRoot Mean Squared Error (RMSE) = {:.4f}\nMean Absolute Error (MAE) = {:.4f}".format(test_rmse, test_mae))
print("\nMean Absolute Percentage Error (MAPE) Over Predictions = {:.4f} %\nCorrelation coefficient (R) = {:.4f}".format(test_mape, test_corr[0][1]))
print("\nMean Training Percentage Bias = {:.4f} %\nMean Test Percentage Bias = {:.4f} %".format(train_p_bias, test_p_bias))

# plot Feature importance
feat_imp = bgb_percentc_model.feature_importances_
sorted_idx = np.argsort(feat_imp)
pos = np.arange(sorted_idx.shape[0]) + 0.5
# Make regression line over y_test and it's predictions
regressor = LinearRegression()
y_test = np.array(y_test).reshape(-1,1)
y_test_pred = np.array(y_test_pred).reshape(-1,1)
regressor.fit(y_test, y_test_pred)
y_pred = regressor.predict(y_test)

def plotFeatureImportance():
    plt.barh(pos, feat_imp[sorted_idx], align="center")
    plt.yticks(pos, np.array(bgb_percentc_model.feature_names_in_)[sorted_idx])
    plt.title("Feature Importance")

plotFeatureImportance()
plotY()

f = open('csv/bgb_carbon_models.pckl', 'wb')
pickle.dump([bgb_percentc_model, bgb_soilc_model], f)
f.close()
