# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""


import os
import ee
import pandas as pd
os.chdir("Code")    # adjust directory

# read csv file and convert dates from strings to datetime
filename = "Belowground Biomass_RS Model.csv"
# REPEAT same for AGB
# filename = "Aboveground Biomass_RS Model.csv"
data = pd.read_csv(filename)
sum(data['Longitude'].isna())
data.head()

# Authenticate and Initialize the Earth Engine API
#ee.Authenticate()
ee.Initialize()


# reads Landsat data, flow accumulation, gridmet temperature and DEM data (for slope and elevation)
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
dem = ee.Image('USGS/SRTMGL1_003').select('elevation')


# add B5 (NIR) value explicitly to properties of landsat
def addB5(image):
    return image.set('B5_value', image.select('SR_B5').reduceRegion(ee.Reducer.mean(), point, 30).get('SR_B5'))


# extract unique years and create a dictionary of landsat data for each year
years = set(x[:4] for x in data.loc[:, 'SampleDate'])
landsat = {}
for year in years:
    landsat[year] = landsat8_collection.filterDate(year+"-01-01", year+"-12-31")


Blue, Green, Red, NIR, SWIR_1, SWIR_2 = [], [], [], [], [], []
flow, slope, elevation = [], [], []
min_temp, max_temp, peak_dates = [], [], []
# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    year = data.loc[idx, 'SampleDate'][:4]
    target_date = data.loc[idx, 'SampleDate']

    # compute min and max temperature from gridmet (resolution = 4,638.3m)
    gridmet_filtered = gridmet.filterBounds(point).filterDate(ee.Date(target_date).advance(-1, 'day'), ee.Date(target_date).advance(1, 'day'))
    bands = ee.Image(gridmet_filtered.first()).select(['tmmn', 'tmmx'])
    temperature_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    tmin = temperature_values['tmmn']
    tmax = temperature_values['tmmx']
    
    # compute flow accumulation (463.83m resolution); slope and aspect (30m resolution)
    flow_value = flow_acc.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    elev = dem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
    slopeDem = ee.Terrain.slope(dem)
    slope_value = slopeDem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
    
    # filter landsat by location and year, and sort by NIR (B5) then extract band values
    spatial_filtered_with_b5 = landsat[year].filterBounds(point).map(addB5)
    peak_sorted_image = spatial_filtered_with_b5.sort('B5_value', False).first()
    bands = peak_sorted_image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    band_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    band_values = list(band_values.values())
    peak_day = peak_sorted_image.getInfo()['properties']['DATE_ACQUIRED']
    
    Blue.append(band_values[0])
    Green.append(band_values[1])
    Red.append(band_values[2])
    NIR.append(band_values[3])
    SWIR_1.append(band_values[4])
    SWIR_2.append(band_values[5])
    flow.append(flow_value)
    elevation.append(elev)
    slope.append(slope_value)
    min_temp.append(tmin)
    max_temp.append(tmax)
    peak_dates.append(peak_day)
    
    if idx%50 == 0: print(idx, end=' ')

# checks if they are all cloud free (should equal data.shape[0])
ids = [x for x in NIR if x]
len(ids)

data['Blue'] = Blue
data['Green'] = Green
data['Red'] = Red
data['NIR'] = NIR
data['SWIR_1'] = SWIR_1
data['SWIR_2'] = SWIR_2
data['Flow'] = flow
data['Elevation'] = elevation
data['Slope'] = slope
data['Minimum_temperature'] = min_temp
data['Maximum_temperature'] = max_temp
data['peak_date'] = peak_dates
data.head()

# write updated dataframe to new csv file
data.to_csv(filename.split(".csv")[0] + "_Data.csv", index=False)


# ML training starts here
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import numpy as np
import math

# read csv containing random samples
data = pd.read_csv("Belowground Biomass_RS Model_Data.csv")
data.head()
data['NDVI'] = (data['NIR'] - data['Red'])/(data['NIR'] + data['Red'])
data['NDWI'] = (data['Green'] - data['NIR'])/(data['Green'] + data['NIR'])
data['EVI'] = 2.5*(data['NIR'] - data['Red'])/(data['NIR'] + 6*data['Red'] - 7.5*data['Blue'] + 1)
data['SAVI'] = 1.5*(data['NIR'] - data['Red'])/(data['NIR'] + data['Red'] + 0.5)
data['BSI'] = ((data['Red'] + data['SWIR_1']) - (data['NIR'] + data['Red']))/(data['Red'] + data['SWIR_1'] + data['NIR'] + data['Red'])
cols = data.columns

# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in cols[6:] if c not in ['percentC', '...1', 'peak_date']]
X = data.loc[:, var_col[3:]]
y_field = 'Roots.kg.m2'
Y = data.loc[:, y_field]

# check for missing/null values across columns and rows respectively (confirm results below should be all 0)
sum(X.isnull().any(axis=0) == True)
sum(X.isnull().any(axis=1) == True)
sum(Y.isnull())

# if NAs where found (results are not 0) in one of them (e.g. Y)
nullIds =  list(np.where(Y.isnull())[0])    # null IDs
X.drop(nullIds, inplace = True)
Y.drop(nullIds, inplace = True)

# split X and Y into training (80%) and test data (20%), random state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

''' Before using gradient boosting, optimize hyperparameters either:
    by randomnly selecting from a range of values using RandomizedSearchCV,
    or by grid search which searches through all provided possibilities with GridSearchCV
'''
# optimize hyperparameters with RandomizedSearchCV on the training data (takes 30ish minutes)
parameters = {'learning_rate': uniform(), 'subsample': uniform(), 
              'n_estimators': randint(5, 5000), 'max_depth': randint(2, 10)}
randm = RandomizedSearchCV(estimator=GradientBoostingRegressor(), param_distributions=parameters,
                           cv=5, n_iter=20,  scoring='neg_mean_squared_error')
randm.fit(X_train, y_train)
randm.best_params_      # outputs all parameters of ideal estimator

# same process above but with GridSearchCV for comparison (takes even longer)
parameters = {'learning_rate': [0.01, 0.03, 0.05], 'subsample': [0.4, 0.5, 0.6, 0.7], 
              'n_estimators': [1000, 2500, 5000], 'max_depth': [6,7,8]}
grid = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=parameters, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
grid.best_params_

gbm_model = GradientBoostingRegressor(learning_rate=0.08, max_depth=8, n_estimators=2359, subsample=0.9,
                                       validation_fraction=0.2, n_iter_no_change=20, max_features='log2',
                                       verbose=1, random_state=48)
gbm_model.fit(X_train, y_train)
len(gbm_model.estimators_)  # number of trees used in estimation

# print relevant stats
y_train_pred = gbm_model.predict(X_train)
y_test_pred = gbm_model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
val = (y_train_pred - y_train) / y_train
train_p_bias = np.mean(val[np.isfinite(val)]) * 100
train_corr = np.corrcoef(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
test_corr = np.corrcoef(y_test, y_test_pred)
val = (y_test_pred - y_test) / y_test
test_p_bias = sum(val[np.isfinite(val)]) * 100

print("TRAINING DATA:\nRoot Mean Squared Error (RMSE) = {}\nMean Absolute Error (MAE) = {}".format(train_rmse, train_mae))
print("\nMean Absolute Percentage Error (MAPE) = {} %\nCorrelation coefficient matrix (R) = {}".format(train_mape, train_corr[0][1]))
print("\nTEST DATA:\nRoot Mean Squared Error (RMSE) = {}\nMean Absolute Error (MAE) = {}".format(test_rmse, test_mae))
print("\nMean Absolute Percentage Error (MAPE) = {} %\nCorrelation coefficient (R) = {}".format(test_mape, test_corr[0][1]))
print("\nMean Training Percentage Bias = {} %\nMean Test Percentage Bias = {} %".format(train_p_bias, test_p_bias))

# plot Feature importance
feat_imp = gbm_model.feature_importances_
sorted_idx = np.argsort(feat_imp)
pos = np.arange(sorted_idx.shape[0]) + 0.5
def plotFeatureImportance():
    plt.barh(pos, feat_imp[sorted_idx], align="center")
    plt.yticks(pos, np.array(gbm_model.feature_names_in_)[sorted_idx])
    plt.title("Feature Importance")
plotFeatureImportance()

def plotY():
    plt.scatter(y_test, y_test_pred, color='g')
    plt.xlabel('Actual ' + y_field)
    plt.ylabel("Predicted " + y_field)
    axes_lim = math.ceil(max(max(y_test), max(y_test_pred))) + 2
    plt.xlim((0, axes_lim))
    plt.ylim((0, axes_lim))
plotY()



# same procedure as above
data = pd.read_csv("Aboveground Biomass_RS Model_Data.csv")
data.head()
data['NDVI'] = (data['NIR'] - data['Red'])/(data['NIR'] + data['Red'])
data['NDWI'] = (data['Green'] - data['NIR'])/(data['Green'] + data['NIR'])
data['EVI'] = 2.5*(data['NIR'] - data['Red'])/(data['NIR'] + 6*data['Red'] - 7.5*data['Blue'] + 1)
data['SAVI'] = 1.5*(data['NIR'] - data['Red'])/(data['NIR'] + data['Red'] + 0.5)
data['BSI'] = ((data['Red'] + data['SWIR_1']) - (data['NIR'] + data['Red']))/(data['Red'] + data['SWIR_1'] + data['NIR'] + data['Red'])
cols = data.columns

# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in cols[5:] if c != 'peak_date']
X = data.loc[:, var_col[2:]]
y_field = 'HerbBio.g.m2'
Y = data.loc[:, y_field]

# check for missing/null values across columns and rows respectively (confirm results below should be all 0)
sum(X.isnull().any(axis=0) == True)
sum(X.isnull().any(axis=1) == True)
sum(Y.isnull())

# if NAs where found (results are not 0) in one of them (e.g. Y)
nullIds =  list(np.where(Y.isnull())[0])    # null IDs
X.drop(nullIds, inplace = True)
Y.drop(nullIds, inplace = True)

# split X and Y into training (80%) and test data (20%), random state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

''' Before using gradient boosting, optimize hyperparameters either:
    by randomnly selecting from a range of values using RandomizedSearchCV,
    or by grid search which searches through all provided possibilities with GridSearchCV
'''
# optimize hyperparameters with RandomizedSearchCV on the training data (takes 30ish minutes)
parameters = {'learning_rate': uniform(), 'subsample': uniform(), 
              'n_estimators': randint(5, 5000), 'max_depth': randint(2, 10)}
randm = RandomizedSearchCV(estimator=GradientBoostingRegressor(), param_distributions=parameters,
                           cv=5, n_iter=50, scoring='neg_mean_squared_error')
randm.fit(X_train, y_train)
randm.best_params_      # outputs all parameters of ideal estimator

# same process above but with GridSearchCV for comparison (takes even longer)
parameters = {'learning_rate': [0.001, 0.003, 0.007, 0.01, 0.02, 0.03], 'subsample': [0.4, 0.5, 0.6, 0.7], 
              'n_estimators': [100, 250, 500, 750, 1000, 2500, 4000], 'max_depth': [3,4,5,6,7,8]}
grid = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=parameters, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
grid.best_params_

gbm_model = GradientBoostingRegressor(learning_rate=0.01, max_depth=6, n_estimators=1000, subsample=0.4,
                                       validation_fraction=0.2, n_iter_no_change=50, max_features='log2',
                                       verbose=1, random_state=48)
gbm_model.fit(X_train, y_train)
len(gbm_model.estimators_)  # number of trees used in estimation

# print relevant stats
y_train_pred = gbm_model.predict(X_train)
y_test_pred = gbm_model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
val = (y_train_pred - y_train) / y_train
train_p_bias = np.mean(val[np.isfinite(val)]) * 100
train_corr = np.corrcoef(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
test_corr = np.corrcoef(y_test, y_test_pred)
val = (y_test_pred - y_test) / y_test
test_p_bias = np.mean(val[np.isfinite(val)]) * 100

print("TRAINING DATA:\nRoot Mean Squared Error (RMSE) = {}\nMean Absolute Error (MAE) = {}".format(train_rmse, train_mae))
print("\nMean Absolute Percentage Error (MAPE) = {} %\nCorrelation coefficient matrix (R) = {}".format(train_mape, train_corr[0][1]))
print("\nTEST DATA:\nRoot Mean Squared Error (RMSE) = {}\nMean Absolute Error (MAE) = {}".format(test_rmse, test_mae))
print("\nMean Absolute Percentage Error (MAPE) = {} %\nCorrelation coefficient (R) = {}".format(test_mape, test_corr[0][1]))
print("\nMean Training Percentage Bias = {} %\nMean Test Percentage Bias = {} %".format(train_p_bias, test_p_bias))

# plot Feature importance
feat_imp = gbm_model.feature_importances_
sorted_idx = np.argsort(feat_imp)
pos = np.arange(sorted_idx.shape[0]) + 0.5
def plotFeatureImportance():
    plt.barh(pos, feat_imp[sorted_idx], align="center")
    plt.yticks(pos, np.array(gbm_model.feature_names_in_)[sorted_idx])
    plt.title("Feature Importance")
plotFeatureImportance()

def plotY():
    plt.scatter(y_test, y_test_pred, color='g')
    plt.xlabel('Actual ' + y_field)
    plt.ylabel("Predicted " + y_field)
    axes_lim = math.ceil(max(max(y_test), max(y_test_pred))) + 2
    plt.xlim((0, axes_lim))
    plt.ylim((0, axes_lim))
plotY()