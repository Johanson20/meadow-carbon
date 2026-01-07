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

mydir = "Code"      # adjust directory
os.chdir(mydir)

# read csv file and convert dates from strings to datetime
filename = "csv/Belowground Biomass_RS Model.csv"
data = pd.read_csv(filename)
data.head()
data.drop_duplicates(inplace=True)  # remove duplicate rows

data.loc[:, ['Longitude', 'Latitude', 'SampleDate']].isna().sum()   # should be 0 for all columns
nullIds =  data[data[['Longitude', 'Latitude', 'SampleDate']].isna().any(axis=1)].index    # rows with null coordinates/dates
data.drop(nullIds, inplace = True)
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
    ndwi = image.normalizedDifference(['NIR', 'SWIR_1']).rename('NDWI')
    evi = image.expression("2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'BLUE': image.select('Blue')}).rename('EVI')
    savi = image.expression("1.5 * ((NIR - RED) / (NIR + RED + 0.5))", {'NIR': image.select('NIR'), 'RED': image.select('Red')}).rename('SAVI')
    bsi = image.expression("((RED + SWIR_1) - (NIR + BLUE)) / (RED + SWIR_1 + NIR + BLUE)", {'RED': image.select('Red'), 'SWIR_1': image.select('SWIR_1'), 'NIR': image.select('NIR'), 'BLUE': image.select('Blue')}).rename('BSI')
    ndpi = image.expression("(NIR - ((0.56 * RED) + (0.44 * SWIR_2))) / (NIR + ((0.56 * RED) + (0.44 * SWIR_2)))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'SWIR_2': image.select('SWIR_2')}).rename('NDPI')
    
    return image.addBands([ndvi, ndwi, evi, savi, bsi, ndpi])


def maskCloud(image):
    # rename bands and mask out cloud based on bits in QA_pixel
    image = image.rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'QA'])
    qa = image.select('QA')
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = dilated_cloud.And(cirrus).And(cloud).And(cloudShadow)
    return image.updateMask(cloud_mask)


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
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1').resample('bilinear').reproject(crs="EPSG:32610", scale=30)
dem = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32610", scale=30)
slopeDem = ee.Terrain.slope(dem)
terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(['tmmn', 'tmmx', 'pr']).map(resample10)

flow_acc_11 = ee.Image("WWF/HydroSHEDS/15ACC").select('b1').resample('bilinear').reproject(crs="EPSG:32611", scale=30)
dem_11 = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32611", scale=30)
slopeDem_11 = ee.Terrain.slope(dem_11)
terraclimate_11 = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(['tmmn', 'tmmx', 'pr']).map(resample11)

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

Blue_Summer, Green_Summer, Red_Summer, NIR_Summer, SWIR_1_Summer, SWIR_2_Summer = [], [], [], [], [], []
NDVI_Summer, NDWI_Summer, EVI_Summer, SAVI_Summer, BSI_Summer, NDPI_Summer = [], [], [], [], [], []
Blue_Fall, Green_Fall, Red_Fall, NIR_Fall, SWIR_1_Fall, SWIR_2_Fall = [], [], [], [], [], []
NDVI_Fall, NDWI_Fall, EVI_Fall, SAVI_Fall, BSI_Fall, NDPI_Fall = [], [], [], [], [], []
Flow, Slope, Elevation, MAP, MAT = [], [], [], [], []
Shallow_Clay, Shallow_Hydra, Shallow_Sand, Lithology, Deep_Clay, Deep_Hydra, Deep_Sand = [], [], [], [], [], [], []

# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    year, month, day = target_date.split("-")
    prev_5_year, prev_30_year = str(int(year)-5) + "-01-01", str(int(year)-30) + "-01-01"
    if int(year) > 2023:    # 2024 data still seems unavailable
        data.drop(idx, inplace=True)
        print("Row", idx, "dropped!")
        continue
    
    # compute 5 year average of landsat bands/indices in June and September
    June_landsat = calculateIndices(landsat_June.filterBounds(point).filterDate(prev_5_year, year+"-12-31").mean())
    Sept_landsat = calculateIndices(landsat_Sept.filterBounds(point).filterDate(prev_5_year, year+"-12-31").mean())
    bands_June = June_landsat.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30).getInfo()
    bands_Sept = Sept_landsat.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30).getInfo()
    
    # compute values from daymetv4 (1km resolution) and gridmet/terraclimate (resolution of both is 4,638.3m)
    if x >= -120:   # latitudes between 120W and 114W refer to EPSG:32611"
        # MAT and MAP values are over 30 years
        pr_values = terraclimate_11.filterBounds(point).filterDate(prev_30_year, year+"-12-31").sum()
        temp_values = terraclimate_11.filterBounds(point).filterDate(prev_30_year, year+"-12-31").mean()
        elev = dem_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
        slope_value = slopeDem_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
        flow_value = flow_acc_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        shallow_clay = shallow_perc_clay_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        deep_clay = deep_perc_clay_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        shallow_hydra = shallow_hydra_cond_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        deep_hydra = deep_hydra_cond_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        shallow_sand = shallow_perc_sand_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        deep_sand = deep_perc_sand_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        lith = lithology_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    else:
        pr_values = terraclimate.filterBounds(point).filterDate(prev_30_year, year+"-12-31").sum()
        temp_values = terraclimate.filterBounds(point).filterDate(prev_30_year, year+"-12-31").mean()
        elev = dem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
        slope_value = slopeDem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
        flow_value = flow_acc.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        shallow_clay = shallow_perc_clay.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        deep_clay = deep_perc_clay.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        shallow_hydra = shallow_hydra_cond.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        deep_hydra = deep_hydra_cond.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        shallow_sand = shallow_perc_sand.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        deep_sand = deep_perc_sand.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        lith = lithology.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        
    sum_pr_values = pr_values.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30).getInfo()
    sum_temp_values = temp_values.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30).getInfo()
    mean_pr = sum_pr_values['pr']/30
    mean_temp = np.mean(list(sum_temp_values.values())[1:])*0.1
    
    Blue_Summer.append(bands_June['Blue'])
    Green_Summer.append(bands_June['Green'])
    Red_Summer.append(bands_June['Red'])
    NIR_Summer.append(bands_June['NIR'])
    SWIR_1_Summer.append(bands_June['SWIR_1'])
    SWIR_2_Summer.append(bands_June['SWIR_2'])
    NDVI_Summer.append(bands_June['NDVI'])
    NDWI_Summer.append(bands_June['NDWI'])
    EVI_Summer.append(bands_June['EVI'])
    SAVI_Summer.append(bands_June['SAVI'])
    BSI_Summer.append(bands_June['BSI'])
    NDPI_Summer.append(bands_June['NDPI'])
    
    Blue_Fall.append(bands_Sept['Blue'])
    Green_Fall.append(bands_Sept['Green'])
    Red_Fall.append(bands_Sept['Red'])
    NIR_Fall.append(bands_Sept['NIR'])
    SWIR_1_Fall.append(bands_Sept['SWIR_1'])
    SWIR_2_Fall.append(bands_Sept['SWIR_2'])
    NDVI_Fall.append(bands_Sept['NDVI'])
    NDWI_Fall.append(bands_Sept['NDWI'])
    EVI_Fall.append(bands_Sept['EVI'])
    SAVI_Fall.append(bands_Sept['SAVI'])
    BSI_Fall.append(bands_Sept['BSI'])
    NDPI_Fall.append(bands_Sept['NDPI'])
    
    MAP.append(mean_pr)
    MAT.append(mean_temp)
    Flow.append(flow_value)
    Elevation.append(elev)
    Slope.append(slope_value)
    Shallow_Clay.append(shallow_clay)
    Shallow_Sand.append(shallow_sand)
    Shallow_Hydra.append(shallow_hydra)
    Lithology.append(lith)
    Deep_Clay.append(deep_clay)
    Deep_Sand.append(deep_sand)
    Deep_Hydra.append(deep_hydra)
    
    if idx%50 == 0: print(idx, end=' ')

# checks if they are all cloud free (should equal data.shape[0])
len([x for x in MAT if x])

data['Blue_June'] = Blue_Summer
data['Green_June'] = Green_Summer
data['Red_June'] = Red_Summer
data['NIR_June'] = NIR_Summer
data['SWIR_1_June'] = SWIR_1_Summer
data['SWIR_2_June'] = SWIR_2_Summer
data['Blue_Sept'] = Blue_Fall
data['Green_Sept'] = Green_Fall
data['Red_Sept'] = Red_Fall
data['NIR_Sept'] = NIR_Fall
data['SWIR_1_Sept'] = SWIR_1_Fall
data['SWIR_2_Sept'] = SWIR_2_Fall

data['Elevation'] = Elevation
data['Flow'] = Flow
data['Slope'] = Slope
data['MAP_30year'] = MAP
data['MAT_30_year'] = MAT

data['NDVI_June'] = NDVI_Summer
data['NDWI_June'] = NDWI_Summer
data['EVI_June'] = EVI_Summer
data['SAVI_June'] = SAVI_Summer
data['BSI_June'] = BSI_Summer
data['NDPI_June'] = NDPI_Summer
data['NDVI_Sept'] = NDVI_Fall
data['NDWI_Sept'] = NDWI_Fall
data['EVI_Sept'] = EVI_Fall
data['SAVI_Sept'] = SAVI_Fall
data['BSI_Sept'] = BSI_Fall
data['NDPI_Sept'] = NDPI_Fall

data['Shallow_Clay'] = Shallow_Clay
data['Deep_Clay'] = Deep_Clay
data['Shallow_Sand'] = Shallow_Sand
data['Deep_Sand'] = Deep_Sand
data['Shallow_Hydra_Conduc'] = Shallow_Hydra
data['Deep_Hydra_Conduc'] = Deep_Hydra
data['Lithology'] = Lithology

data.reset_index(drop=True, inplace=True)
data.head()

# write updated dataframe to new csv file
data.to_csv(filename.split(".csv")[0] + "_Carbon_Data.csv", index=False)


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
data = pd.read_csv("csv/Belowground Biomass_RS Model_5_year_Data.csv")
data.head()
cols = data.columns
# cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)

# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in list(cols[10:]) if c not in ['dNDSI', 'Cdef', 'Flow', 'Lithology', 'Organic_Matter', 'Snow_days', 'Minimum_temperature', 'Maximum_temperature']]
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
for value in range(2, 16, 2):   # max rounded value is 16
    mask = (data[y_field] > value) & (data[y_field] <= (value+2))
    data.loc[mask, 'SoilC'] = value//2
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

bgb_soilc_model = GradientBoostingRegressor(learning_rate=0.16, max_depth=9, n_estimators=25, subsample=0.7,
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
data = pd.read_csv("csv/Belowground Biomass_RS Model_5_year_Data.csv")
data.head()
cols = data.columns
# cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)

# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in list(cols[10:]) if c not in ['dNDSI', 'Cdef', 'Flow', 'Lithology', 'Organic_Matter', 'Snow_days', 'Minimum_temperature', 'Maximum_temperature']]
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
for value in range(2, 16, 2):   # max rounded value is 16
    mask = (data[y_field] > value) & (data[y_field] <= (value+2))
    data.loc[mask, 'percentCarbon'] = value//2
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

bgb_percentc_model = GradientBoostingRegressor(learning_rate=0.3, max_depth=4, n_estimators=25, subsample=0.7,
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
