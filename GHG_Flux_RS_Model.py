# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""


import os
import ee
import pandas as pd

mydir = "Code"      # adjust directory
os.chdir(mydir)

# read csv file and convert dates from strings to datetime
data = pd.read_csv("csv/GHG Flux_RS Model.csv")
sum(data['Longitude'].isna())
data.head()

# Authenticate and Initialize the Earth Engine API
#ee.Authenticate()
ee.Initialize()


# reads Landsat data, flow accumulation, gridmet temperature and DEM data (for slope and elevation)
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
landsat7_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
dem = ee.Image('USGS/3DEP/10m').select('elevation')
slopeDem = ee.Terrain.slope(dem)

def maskClouds(image):
    # mask out cloud based on bits in QA_pixel
    qa = image.select('QA_PIXEL')
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    # update image with combined mask
    cloud_mask = dilated_cloud.And(cloud).And(cloudShadow)
    return image.updateMask(cloud_mask)

# Calculates absolute time difference (in days) from a target date, in which the images are acquired
def calculate_time_difference(image):
    time_difference = ee.Number(image.date().difference(target_date, 'day')).abs()
    return image.set('time_difference', time_difference)

def date_in_previous_year(today):
    curr_date = today.split("-")
    curr_date[0] = str(int(curr_date[0]) - 1)
    return "-".join(curr_date)

# Function to extract cloud free band values per pixel from landsat 8 or landsat 7
def getBandValues(landsat_collection, point, target_date, bufferDays = 60, landsatNo = 8):
    # filter landsat images by location
    spatial_filtered = landsat_collection.filterBounds(point)
    # filter the streamlined images by dates +/- a certain number of days
    temporal_filtered = spatial_filtered.filterDate(ee.Date(target_date).advance(-bufferDays, 'day'), ee.Date(target_date).advance(bufferDays, 'day'))
    # apply cloud mask and sort images in the collection
    cloud_free_images = temporal_filtered.map(maskClouds)
    # Map the ImageCollection over time difference and sort to get image of closest date
    sorted_collection = cloud_free_images.map(calculate_time_difference).sort('time_difference')
    image_list = sorted_collection.toList(sorted_collection.size())
    noImages = image_list.size().getInfo()
    nImage, band_values = 0, {'SR_B2': None}
    
    # repeatedly check for cloud free pixels (non-null value) in landsat 8, or checks in landsat 7
    while band_values['SR_B2'] == None and nImage < noImages:
        nearest_image = ee.Image(image_list.get(nImage))
        nImage += 1
        if landsatNo == 7:
            bands = nearest_image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])
        else:
            bands = nearest_image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
        time_diff = nearest_image.getInfo()['properties']['time_difference']
        band_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    
    return [list(band_values.values()), time_diff, nearest_image.projection()]


# define arrays to store band values and landsat information
Blue, Green, Red, NIR, SWIR_1, SWIR_2 = [], [], [], [], [], []
prev_Green, prev_Red, prev_NIR = [], [], []
flow, slope, elevation, driver, time_diff = [], [], [], [], []
min_temp, max_temp = [], []

# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    prev_year_date = date_in_previous_year(target_date)
    
    # extract Landsat band values
    vxn = 8
    band_values, t_diff, mycrs = getBandValues(landsat8_collection, point, target_date)
    prev_band_values, _, _ = getBandValues(landsat8_collection, point, prev_year_date)
    if not band_values[0]:
        vxn = 7
        print(idx, "Searching Landsat 7 collection (60-day search radius)")
        band_values, t_diff, mycrs = getBandValues(landsat7_collection, point, target_date, 60, 7)
    if not prev_band_values[0]:
        print(idx, "Searching Landsat 7 collection for previous year")
        prev_band_values, _, _ = getBandValues(landsat7_collection, point, prev_year_date, 60, 7)
    
    # compute min and max temperature from gridmet (resolution = 4,638.3m)
    gridmet_filtered = gridmet.filterBounds(point).filterDate(ee.Date(target_date).advance(-1, 'day'), ee.Date(target_date).advance(1, 'day'))
    gridmet_30m = gridmet_filtered.first().resample('bilinear').reproject(crs=mycrs, scale=30).select(['tmmn', 'tmmx'])
    temperature_values = gridmet_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    tmin = temperature_values['tmmn']
    tmax = temperature_values['tmmx']
    
    # compute flow accumulation (463.83m resolution); slope and aspect (10.2m resolution)
    flow_30m = flow_acc.resample('bilinear').reproject(crs=mycrs, scale=30)
    dem_30m = dem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear').reproject(crs=mycrs, scale=30)
    slope_30m = slopeDem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear').reproject(crs=mycrs, scale=30)
    flow_value = flow_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    elev = dem_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
    slope_value = slope_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
    
    # append vaues to different lists
    Blue.append(band_values[0])
    Green.append(band_values[1])
    Red.append(band_values[2])
    NIR.append(band_values[3])
    SWIR_1.append(band_values[4])
    SWIR_2.append(band_values[5])
    
    prev_Green.append(prev_band_values[1])
    prev_Red.append(prev_band_values[2])
    prev_NIR.append(prev_band_values[3])
    
    flow.append(flow_value)
    elevation.append(elev)
    slope.append(slope_value)
    driver.append(vxn)
    time_diff.append(t_diff)
    min_temp.append(tmin)
    max_temp.append(tmax)

    if idx%100 == 0: print(idx, end=' ')


data['prev_Green'] = [(x*2.75e-05 - 0.2) for x in prev_Green]
data['prev_Red'] = [(x*2.75e-05 - 0.2) for x in prev_Red]
data['prev_NIR'] = [(x*2.75e-05 - 0.2) for x in prev_NIR]
data['prev_NDVI'] = (data['prev_NIR'] - data['prev_Red'])/(data['prev_NIR'] + data['prev_Red'])
data['prev_NDWI'] = (data['prev_Green'] - data['prev_NIR'])/(data['prev_Green'] + data['prev_NIR'])

data['Blue'] = [(x*2.75e-05 - 0.2) for x in Blue]
data['Green'] = [(x*2.75e-05 - 0.2) for x in Green]
data['Red'] = [(x*2.75e-05 - 0.2) for x in Red]
data['NIR'] = [(x*2.75e-05 - 0.2) for x in NIR]
data['SWIR_1'] = [(x*2.75e-05 - 0.2) for x in SWIR_1]
data['SWIR_2'] = [(x*2.75e-05 - 0.2) for x in SWIR_2]

data['Flow'] = flow
data['Elevation'] = elevation
data['Slope'] = slope
data['Driver'] = driver
data['Days_of_data_acquisition_offset'] = time_diff
data['Minimum_temperature'] = min_temp
data['Maximum_temperature'] = max_temp

data['NDVI'] = (data['NIR'] - data['Red'])/(data['NIR'] + data['Red'])
data['NDWI'] = (data['Green'] - data['NIR'])/(data['Green'] + data['NIR'])
data['EVI'] = 2.5*(data['NIR'] - data['Red'])/(data['NIR'] + 6*data['Red'] - 7.5*data['Blue'] + 1)
data['SAVI'] = 1.5*(data['NIR'] - data['Red'])/(data['NIR'] + data['Red'] + 0.5)
data['BSI'] = ((data['Red'] + data['SWIR_1']) - (data['NIR'] + data['Red']))/(data['Red'] + data['SWIR_1'] + data['NIR'] + data['Red'])

# display first 10 rows of updated dataframe
data.head(10)

# checks how many pixels are cloud free (non-null value);
# all bands would be simultaneously cloud-free or not
ids = [x for x in Blue if x]
len(ids)

# write updated dataframe to new csv file
data.to_csv('csv/GHG_Flux_RS_Model_Data.csv', index=False)


# ML training starts here
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import numpy as np

# read csv containing random samples
data = pd.read_csv("csv/GHG_Flux_RS_Model_Data.csv")
data.head()
# confirm column names first
# cols = data.columns
cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)  # remove duplicate rows
data['ID'].value_counts()

# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in cols[13:] if c not in ['Driver', 'Days_of_data_acquisition_offset']]
y_field = 'CO2.umol.m2.s'
# subdata excludes other measured values which can be largely missing (as we need to assess just one output at a time)
subdata = data.loc[:, ([y_field] + var_col)]

# check for missing/null values across columns and rows respectively (confirm results below should be all 0)
sum(subdata.isnull().any(axis=0) == True)
sum(subdata[y_field].isnull())

# if NAs where found (results above are not 0) in one of them (e.g. Y)
nullIds =  list(np.where(subdata[y_field].isnull())[0])    # null IDs
data.drop(nullIds, inplace = True)
data.dropna(subset=[y_field], inplace=True)
data.reset_index(drop=True, inplace=True)

# split data into training (80%) and test data (20%) by IDs, random state ensures reproducibility
gsp = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=10)
split = gsp.split(data, groups=data['ID'])
train_index, test_index = next(split)
train_data = data.iloc[train_index]
test_data = data.iloc[test_index]

X_train, y_train = train_data.loc[:, var_col], train_data[y_field]
X_test, y_test = test_data.loc[:, var_col], test_data[y_field]

''' Before using gradient boosting, optimize hyperparameters either:
    by randomnly selecting from a range of values using RandomizedSearchCV,
    or by grid search which searches through all provided possibilities with GridSearchCV
'''
# optimize hyperparameters with RandomizedSearchCV on the training data (takes 30ish minutes)
parameters = {'learning_rate': uniform(), 'subsample': uniform(), 
              'n_estimators': randint(5, 5000), 'max_depth': randint(2, 10)}
randm = RandomizedSearchCV(estimator=GradientBoostingRegressor(), param_distributions=parameters,
                           cv=5, n_iter=20, scoring='neg_mean_squared_error')
randm.fit(X_train, y_train)
randm.best_params_      # outputs all parameters of ideal estimator

# same process above but with GridSearchCV for comparison (takes even longer)
parameters = {'learning_rate': [0.03], 'subsample': [0.5, 0.6], 
              'n_estimators': [2500], 'max_depth': [4,5,6,7,8,9]}
grid = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=parameters, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
grid.best_params_

# run gradient boosting with optimized parameters (chosen with GridSearchCV) on training data
ghg_model = GradientBoostingRegressor(learning_rate=0.03, max_depth=11, n_estimators=250, subsample=0.8,
                                       validation_fraction=0.1, n_iter_no_change=50, max_features='log2',
                                       verbose=1, random_state=48)
ghg_model.fit(X_train, y_train)
len(ghg_model.estimators_)  # number of trees used in estimation

# print relevant stats
y_train_pred = ghg_model.predict(X_train)
y_test_pred = ghg_model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mape = mean_absolute_percentage_error(y_train_pred, y_train)
val = (y_train_pred - y_train) / y_train
train_p_bias = np.mean(val[np.isfinite(val)]) * 100
train_corr = np.corrcoef(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test_pred, y_test)
test_corr = np.corrcoef(y_test, y_test_pred)
val = (y_test_pred - y_test) / y_test
test_p_bias = np.mean(val[np.isfinite(val)]) * 100

print("\nTRAINING DATA:\nRoot Mean Squared Error (RMSE) = {}\nMean Absolute Error (MAE) = {}".format(train_rmse, train_mae))
print("\nMean Absolute Percentage Error (MAPE) Over Predictions = {} %\nCorrelation coefficient matrix (R) = {}".format(train_mape, train_corr[0][1]))
print("\nTEST DATA:\nRoot Mean Squared Error (RMSE) = {}\nMean Absolute Error (MAE) = {}".format(test_rmse, test_mae))
print("\nMean Absolute Percentage Error (MAPE) Over Predictions = {} %\nCorrelation coefficient (R) = {}".format(test_mape, test_corr[0][1]))
print("\nMean Training Percentage Bias = {} %\nMean Test Percentage Bias = {} %".format(train_p_bias, test_p_bias))

# plot Feature importance
feat_imp = ghg_model.feature_importances_
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
    plt.yticks(pos, np.array(ghg_model.feature_names_in_)[sorted_idx])
    plt.title("Feature Importance")
plotFeatureImportance()

def plotY():
    plt.scatter(y_test, y_test_pred, color='g')
    plt.plot(y_test, y_pred, color='k', label='Regression line')
    plt.plot(y_test, y_test, linestyle='dotted', color='gray', label='1:1 line')
    plt.xlabel('Actual ' + y_field)
    plt.ylabel("Predicted " + y_field)
    plt.title("Test set (y_test) predictions")
    # Make axes of equal extents
    axes_lim = np.ceil(max(max(y_test), max(y_test_pred))) + 2
    plt.xlim((0, axes_lim))
    plt.ylim((0, axes_lim))
    plt.legend()
plotY()
