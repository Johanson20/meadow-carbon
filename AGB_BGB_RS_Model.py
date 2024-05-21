# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""


import os
import ee
import pickle
import numpy as np
import pandas as pd
os.chdir("Code")    # adjust directory

# read csv file and convert dates from strings to datetime
filename = "csv/Belowground Biomass_RS Model.csv"
# REPEAT same for AGB
# filename = "csv/Aboveground Biomass_RS Model.csv"
data = pd.read_csv(filename)
data.head()

sum(data['Longitude'].isna())
nullIds =  list(np.where(data['Longitude'].isnull())[0])    # rows with null coordinates
data.drop(nullIds, inplace = True)
data.reset_index(drop=True, inplace=True)
# adjust datetime format
data['SampleDate'] = pd.to_datetime(data['SampleDate'], format="%m/%d/%Y")
data['Previous_Year'] = data['SampleDate'] - pd.DateOffset(years=1)
data['Previous_Year'] = data['Previous_Year'].dt.strftime('%Y-%m-%d')
data['SampleDate'] = data['SampleDate'].dt.strftime('%Y-%m-%d')
data.head()

# Authenticate and Initialize the Earth Engine API
#ee.Authenticate()
ee.Initialize()


# reads Landsat data, flow accumulation, gridmet temperature and DEM data (for slope and elevation)
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
dem = ee.Image('USGS/3DEP/10m').select('elevation')
slopeDem = ee.Terrain.slope(dem)


# add B5 (NIR) value explicitly to properties of landsat
def addB5(image):
    return image.set('B5_value', image.select('SR_B5').reduceRegion(ee.Reducer.mean(), point, 30).get('SR_B5'))


# extract unique years and create a dictionary of landsat data for each year
years = set([x[:4] for x in data.loc[:, 'SampleDate']] + [x[:4] for x in data.loc[:, 'Previous_Year']])
landsat = {}
for year in years:
    landsat[year] = landsat8_collection.filterDate(year+"-01-01", year+"-12-31")


Blue, Green, Red, NIR, SWIR_1, SWIR_2 = [], [], [], [], [], []
prev_Blue, prev_Green, prev_Red, prev_NIR, prev_SWIR_1, prev_SWIR_2 = [], [], [], [], [], []
flow, slope, elevation = [], [], []
min_temp, max_temp, peak_dates = [], [], []
# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    year = target_date[:4]
    previous_year = data.loc[idx, 'Previous_Year'][:4]
    
    # filter landsat by location and year, and sort by NIR (B5) then extract band values
    spatial_filtered_with_b5 = landsat[year].filterBounds(point).map(addB5)
    peak_sorted_image = spatial_filtered_with_b5.sort('B5_value', False).first()
    bands = peak_sorted_image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    band_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    band_values = list(band_values.values())
    peak_day = peak_sorted_image.getInfo()['properties']['DATE_ACQUIRED']
    
    # extract same peak landsat band values for previous year
    spatial_filtered_with_b5 = landsat[previous_year].filterBounds(point).map(addB5)
    peak_sorted_image = spatial_filtered_with_b5.sort('B5_value', False).first()
    bands = peak_sorted_image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    prev_band_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    prev_band_values = list(prev_band_values.values())
    
    # compute min and max temperature from gridmet (resolution = 4,638.3m)
    gridmet_filtered = gridmet.filterBounds(point).filterDate(ee.Date(target_date).advance(-1, 'day'), ee.Date(target_date).advance(1, 'day')).first()
    gridmet_30m = gridmet_filtered.resample('bilinear').reproject(crs=peak_sorted_image.projection(), scale=30).select(['tmmn', 'tmmx'])
    temperature_values = gridmet_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    tmin = temperature_values['tmmn']
    tmax = temperature_values['tmmx']
    
    # compute flow accumulation (463.83m resolution); slope and aspect (10.2m resolution)
    flow_30m = flow_acc.resample('bilinear').reproject(crs=peak_sorted_image.projection(), scale=30)
    dem_30m = dem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear').reproject(crs=peak_sorted_image.projection(), scale=30)
    slope_30m = slopeDem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear').reproject(crs=peak_sorted_image.projection(), scale=30)
    flow_value = flow_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    elev = dem_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
    slope_value = slope_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
    
    Blue.append(band_values[0])
    Green.append(band_values[1])
    Red.append(band_values[2])
    NIR.append(band_values[3])
    SWIR_1.append(band_values[4])
    SWIR_2.append(band_values[5])
    
    prev_Blue.append(prev_band_values[0])
    prev_Green.append(prev_band_values[1])
    prev_Red.append(prev_band_values[2])
    prev_NIR.append(prev_band_values[3])
    prev_SWIR_1.append(prev_band_values[4])
    prev_SWIR_2.append(prev_band_values[5])
    
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

data['prev_Blue'] = prev_Blue
data['prev_Green'] = prev_Green
data['prev_Red'] = prev_Red
data['prev_NIR'] = prev_NIR
data['prev_SWIR_1'] = prev_SWIR_1
data['prev_SWIR_2'] = prev_SWIR_2

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
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import numpy as np

# read csv containing random samples
data = pd.read_csv("csv/Belowground Biomass_RS Model_Data.csv")
data.head()
data['NDVI'] = (data['NIR'] - data['Red'])/(data['NIR'] + data['Red'])
data['NDWI'] = (data['Green'] - data['NIR'])/(data['Green'] + data['NIR'])
data['EVI'] = 2.5*(data['NIR'] - data['Red'])/(data['NIR'] + 6*data['Red'] - 7.5*data['Blue'] + 1)
data['SAVI'] = 1.5*(data['NIR'] - data['Red'])/(data['NIR'] + data['Red'] + 0.5)
data['BSI'] = ((data['Red'] + data['SWIR_1']) - (data['NIR'] + data['Red']))/(data['Red'] + data['SWIR_1'] + data['NIR'] + data['Red'])
cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)  # remove duplicate rows
data['ID'].value_counts()

# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in cols[5:] if c not in ['percentC', '...1', 'peak_date']]
y_field = 'Roots.kg.m2'
# subdata excludes other measured values which can be largely missing (as we need to assess just one output at a time)
subdata = data.loc[:, ([y_field] + var_col[3:])]

# check for missing/null values across columns and rows respectively (confirm results below should be all 0)
sum(subdata.isnull().any(axis=0) == True)
sum(subdata[y_field].isnull())

# if NAs where found (results above are not 0) in one of them (e.g. Y)
nullIds =  list(np.where(subdata[y_field].isnull())[0])    # null IDs
data.drop(nullIds, inplace = True)

# split data into training (80%) and test data (20%) by IDs, random state ensures reproducibility
gsp = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=10)
split = gsp.split(data, groups=data['ID'])
train_index, test_index = next(split)
train_data = data.iloc[train_index]
test_data = data.iloc[test_index]

X_train, y_train = train_data.loc[:, var_col[3:]], train_data[y_field]
X_test, y_test = test_data.loc[:, var_col[3:]], test_data[y_field]

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

# defaults outperform these tuned values
bgb_model = GradientBoostingRegressor(learning_rate=0.28, max_depth=9, n_estimators=100, subsample=0.3,
                                       validation_fraction=0.2, n_iter_no_change=50, max_features='log2',
                                       verbose=1, random_state=48)
bgb_model.fit(X_train, y_train)
len(bgb_model.estimators_)  # number of trees used in estimation

# print relevant stats
y_train_pred = bgb_model.predict(X_train)
y_test_pred = bgb_model.predict(X_test)

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
test_p_bias = sum(val[np.isfinite(val)]) * 100

print("\nTRAINING DATA:\nRoot Mean Squared Error (RMSE) = {}\nMean Absolute Error (MAE) = {}".format(train_rmse, train_mae))
print("\nMean Absolute Percentage Error (MAPE) Over Predictions = {} %\nCorrelation coefficient matrix (R) = {}".format(train_mape, train_corr[0][1]))
print("\nTEST DATA:\nRoot Mean Squared Error (RMSE) = {}\nMean Absolute Error (MAE) = {}".format(test_rmse, test_mae))
print("\nMean Absolute Percentage Error (MAPE) Over Predictions = {} %\nCorrelation coefficient (R) = {}".format(test_mape, test_corr[0][1]))
print("\nMean Training Percentage Bias = {} %\nMean Test Percentage Bias = {} %".format(train_p_bias, test_p_bias))

# plot Feature importance
feat_imp = bgb_model.feature_importances_
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
    plt.yticks(pos, np.array(bgb_model.feature_names_in_)[sorted_idx])
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



# same procedure as above
data = pd.read_csv("csv/Aboveground Biomass_RS Model_Data.csv")
data.head()
data['NDVI'] = (data['NIR'] - data['Red'])/(data['NIR'] + data['Red'])
data['NDWI'] = (data['Green'] - data['NIR'])/(data['Green'] + data['NIR'])
data['EVI'] = 2.5*(data['NIR'] - data['Red'])/(data['NIR'] + 6*data['Red'] - 7.5*data['Blue'] + 1)
data['SAVI'] = 1.5*(data['NIR'] - data['Red'])/(data['NIR'] + data['Red'] + 0.5)
data['BSI'] = ((data['Red'] + data['SWIR_1']) - (data['NIR'] + data['Red']))/(data['Red'] + data['SWIR_1'] + data['NIR'] + data['Red'])
cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)  # remove duplicate rows
data['ID'].value_counts()

# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in cols[4:] if c != 'peak_date']
y_field = 'HerbBio.g.m2'
# subdata excludes other measured values which can be largely missing (as we need to assess just one output at a time)
subdata = data.loc[:, ([y_field] + var_col[2:])]

# check for missing/null values across columns and rows respectively (confirm results below should be all 0)
sum(subdata.isnull().any(axis=0) == True)
sum(subdata[y_field].isnull())

# if NAs where found (results above are not 0) in one of them (e.g. Y)
nullIds =  list(np.where(subdata[y_field].isnull())[0])    # null IDs
data.drop(nullIds, inplace = True)

# split data into training (80%) and test data (20%) by IDs, random state ensures reproducibility
gsp = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=10)
split = gsp.split(data, groups=data['ID'])
train_index, test_index = next(split)
train_data = data.iloc[train_index]
test_data = data.iloc[test_index]

X_train, y_train = train_data.loc[:, var_col[2:]], train_data[y_field]
X_test, y_test = test_data.loc[:, var_col[2:]], test_data[y_field]

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

agb_model = GradientBoostingRegressor(learning_rate=0.29, max_depth=6, n_estimators=125, subsample=0.95,
                                       validation_fraction=0.2, n_iter_no_change=50, max_features='log2',
                                       verbose=1, random_state=48)
agb_model.fit(X_train, y_train)
len(agb_model.estimators_)  # number of trees used in estimation

# print relevant stats
y_train_pred = agb_model.predict(X_train)
y_test_pred = agb_model.predict(X_test)

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
feat_imp = agb_model.feature_importances_
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
    plt.yticks(pos, np.array(agb_model.feature_names_in_)[sorted_idx])
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


f = open('files/models.pckl', 'wb')
pickle.dump([ghg_model, agb_model, bgb_model], f)
f.close()


'''
# GHG Results:
{'RMSE': [1.35099, 0.003, 1500, 0.9, 10, 0.1],
 'BIAS': [22.56739, 0.05, 500, 0.8, 10, 0.2]}

# BGB Results:
{'RMSE': [1.3487, 0.25, 75, 0.6, 5, 0.2],
 'BIAS': [3.36418, 0.28, 100, 0.3, 9, 0.2]}

# AGB Results:
{'RMSE': [92.31518, 0.29, 125, 0.95, 6, 0.2],
 'BIAS': [0.01486, 0.33, 75, 0.4, 13, 0.2]}'''