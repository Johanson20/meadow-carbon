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
data.head()
data.drop('Unnamed: 0', axis=1, inplace=True)
data.drop_duplicates(inplace=True)  # remove duplicate rows
data.loc[:, ['Longitude', 'Latitude', 'SampleDate']].isna().sum()   # should be 0 for all columns
nullIds = data[data[['Longitude', 'Latitude', 'SampleDate']].isna().any(axis=1)].index
data.drop(nullIds, inplace = True)
data.reset_index(drop=True, inplace=True)
data['SampleDate'] = pd.to_datetime(data['SampleDate'], format="%m/%d/%Y").dt.strftime('%Y-%m-%d')

# Authenticate and Initialize the Earth Engine API
#ee.Authenticate()
ee.Initialize()


def maskAndRename(image):
    # rename bands and mask out cloud based on bits in QA_pixel
    image = image.rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'QA'])
    qa = image.select('QA')
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = dilated_cloud.And(cirrus).And(cloud).And(cloudShadow)
    return image.updateMask(cloud_mask).select(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2'])

# Calculates absolute time difference (in days) from a target date, in which the images are acquired
def calculate_time_difference(image):
    time_difference = ee.Number(image.date().difference(target_date, 'day')).abs()
    return image.set('time_difference', time_difference)

# Function to extract cloud free band values per pixel from landsat 8 or landsat 7
def getBandValues(landsat_collection, point, target_date, bufferDays = 60):
    # filter landsat images by location and dates about 60 day radius and sort by proximity to sample date
    spatial_filtered = landsat_collection.filterBounds(point)
    temporal_filtered = spatial_filtered.filterDate(ee.Date(target_date).advance(-bufferDays, 'day'), ee.Date(target_date).advance(bufferDays, 'day'))
    # Map the ImageCollection over time difference and sort to get image of closest date
    sorted_collection = temporal_filtered.map(calculate_time_difference).sort('time_difference')
    noImages = sorted_collection.size().getInfo()
    
    if not noImages:
        return [[0], None, None, None]
    
    image_list = sorted_collection.toList(sorted_collection.size())
    nImage, band_values = 0, {'Blue': None}
    
    # repeatedly check for cloud free pixels (non-null value) in landsat 8, or checks in landsat 7
    while band_values['Blue'] == None and nImage < noImages:
        nearest_image = ee.Image(image_list.get(nImage))
        nImage += 1
        value = nearest_image.getInfo()['properties']
        band_values = nearest_image.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    
    return [band_values, value['time_difference'], value['SPACECRAFT_ID'], 'EPSG:326' + str(value['UTM_ZONE'])]


def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=30)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=30)

# reads Landsat data, flow accumulation, gridmet temperature, terraclimate and DEM data (for slope and elevation)
landsat9_collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat7_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
# flow accumulation (463.83m resolution); slope and elevation (10.2m resolution); 
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1').resample('bilinear').reproject(crs="EPSG:32610", scale=30)
dem = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32610", scale=30)
slopeDem = ee.Terrain.slope(dem)
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select(['tmmn', 'tmmx']).map(resample10)
terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select('aet').map(resample10)

flow_acc_11 = ee.Image("WWF/HydroSHEDS/15ACC").select('b1').resample('bilinear').reproject(crs="EPSG:32611", scale=30)
dem_11 = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32611", scale=30)
slopeDem_11 = ee.Terrain.slope(dem_11)
gridmet_11 = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select(['tmmn', 'tmmx']).map(resample11)
terraclimate_11 = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select('aet').map(resample11)

landsat_collection = landsat9_collection.merge(landsat8_collection).merge(landsat7_collection).map(maskAndRename)
# define arrays to store band values and landsat information
Blue, Green, Red, NIR, SWIR_1, SWIR_2 = [], [], [], [], [], []
flow, slope, elevation, driver, time_diff = [], [], [], [], []
min_temp, max_temp, et = [], [], []

# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    start_month, end_month = target_date[:-2] + "01", target_date[:5] + str(int(target_date[5:7])+1) + "-01"
    if int(end_month[5:-3]) > 12:
        end_month = str(int(end_month[:4]) + 1) + "-01-" + end_month[-2:]
    
    # extract Landsat band values
    band_values, t_diff, vxn, mycrs = getBandValues(landsat_collection, point, target_date)
    if not band_values['Blue']:
        data.drop(idx, inplace=True)
        print("Row", idx, "dropped!")
        continue
    
    # compute temperature and AET from gridmet and terraclimate respectively (both have resolution of 4,638.3m)
    if mycrs == "EPSG:32611":
        tclimate = terraclimate_11.filterBounds(point).filterDate(start_month, end_month)
        gridmet_30m = gridmet_11.filterBounds(point).filterDate(ee.Date(target_date).advance(-1, 'day'), ee.Date(target_date).advance(1, 'day')).first()
    else:
        tclimate = terraclimate.filterBounds(point).filterDate(start_month, end_month)
        gridmet_30m = gridmet.filterBounds(point).filterDate(ee.Date(target_date).advance(-1, 'day'), ee.Date(target_date).advance(1, 'day')).first()
    aet = tclimate.first().reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['aet']
    # compute flow accumulation (463.83m resolution); slope and aspect (10.2m resolution)
    if mycrs == "EPSG:32611":
        elev = dem_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
        slope_value = slopeDem_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
        flow_value = flow_acc_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    else:
        elev = dem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
        slope_value = slopeDem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
        flow_value = flow_acc.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    
    temperature_values = gridmet_30m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    tmin = temperature_values['tmmn']
    tmax = temperature_values['tmmx']
    
    # append vaues to different lists
    Blue.append(band_values['Blue']*2.75e-05 - 0.2)
    Green.append(band_values['Green']*2.75e-05 - 0.2)
    Red.append(band_values['Red']*2.75e-05 - 0.2)
    NIR.append(band_values['NIR']*2.75e-05 - 0.2)
    SWIR_1.append(band_values['SWIR_1']*2.75e-05 - 0.2)
    SWIR_2.append(band_values['SWIR_2']*2.75e-05 - 0.2)
    
    flow.append(flow_value)
    elevation.append(elev)
    slope.append(slope_value)
    driver.append(vxn)
    time_diff.append(t_diff)
    min_temp.append(tmin)
    max_temp.append(tmax)
    et.append(aet)

    if idx%100 == 0: print(idx, end=' ')


data['Blue'] = Blue
data['Green'] = Green
data['Red'] = Red
data['NIR'] = NIR
data['SWIR_1'] = SWIR_1
data['SWIR_2'] = SWIR_2

data['Flow'] = flow
data['Elevation'] = elevation
data['Slope'] = slope
data['AET'] = et
data['Driver'] = driver
data['Days_of_data_acquisition_offset'] = time_diff
data['Minimum_temperature'] = min_temp
data['Maximum_temperature'] = max_temp

data['NDVI'] = (data['NIR'] - data['Red'])/(data['NIR'] + data['Red'])
data['NDWI'] = (data['NIR'] - data['SWIR_1'])/(data['NIR'] + data['SWIR_1'])
data['EVI'] = 2.5*(data['NIR'] - data['Red'])/(data['NIR'] + 6*data['Red'] - 7.5*data['Blue'] + 1)
data['SAVI'] = 1.5*(data['NIR'] - data['Red'])/(data['NIR'] + data['Red'] + 0.5)
data['BSI'] = ((data['Red'] + data['SWIR_1']) - (data['NIR'] + data['Blue']))/(data['Red'] + data['SWIR_1'] + data['NIR'] + data['Blue'])
data['NDPI'] = (data['NIR'] - (0.56*data['Red'] + 0.44*data['SWIR_2']))/(data['NIR'] + 0.56*data['Red'] + 0.44*data['SWIR_2'])

# display first 5 rows of updated dataframe
data.reset_index(drop=True, inplace=True)
data.head()

# checks how many pixels are cloud free (non-null value);
# all bands would be simultaneously cloud-free or not
len([x for x in Blue if x])

# write updated dataframe to new csv file
data.to_csv('csv/GHG_Flux_RS_Model_Data.csv', index=False)


# ML training starts here
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import numpy as np

# read csv containing random samples
data = pd.read_csv("csv/GHG_Flux_RS_Model_Data.csv")
data.head()
# confirm column names first
cols = data.columns
# cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)

# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in cols[8:] if c not in ['Elevation', 'Driver', 'Days_of_data_acquisition_offset']]
y_field = 'CO2.umol.m2.s'
# subdata excludes other measured values which can be largely missing (as we need to assess just one output at a time)
subdata = data.loc[:, ([y_field] + var_col)]

# check for missing/null values across columns and rows respectively (confirm results below should be all 0)
sum(subdata.isnull().any(axis=0) == True)
sum(subdata[y_field].isnull())

# drop C02 values less than -1
outlierIds = data[data[y_field] < -1].index
data.drop(outlierIds, inplace = True)
# if NAs where found (results above are not 0) in one of them (e.g. Y)
nullIds =  list(np.where(subdata[y_field].isnull())[0])    # null IDs
data.drop(nullIds, inplace = True)
data.dropna(subset=[y_field], inplace=True)
data.reset_index(drop=True, inplace=True)
# make scatter plots of relevant variables from raw dataframe
with PdfPages('files/GHG_Scatter_plots.pdf') as pdf:
    for feature in var_col:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x=feature, y=y_field, data=data, line_kws={"color":"red"}, ax=ax)
        ax.set_title(f'Scatter plot of {feature} vs {y_field}')
        pdf.savefig(fig)
        plt.close(fig)

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
ghg_model = GradientBoostingRegressor(learning_rate=0.03, max_depth=6, n_estimators=750, subsample=0.6, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
ghg_84_model = GradientBoostingRegressor(loss="quantile", learning_rate=0.03, alpha=0.8413, max_depth=6, 
                                      n_estimators=750, subsample=0.6, validation_fraction=0.2, n_iter_no_change=50,  
                                      max_features='log2', random_state=10)
ghg_model.fit(X_train, y_train)
ghg_84_model.fit(X_train, y_train)
# Make partial dependence plots
with PdfPages('files/GHG_partial_dependence_plots.pdf') as pdf:
    for i in range(len(var_col)):
        fig, ax = plt.subplots(figsize=(8, 6))
        PartialDependenceDisplay.from_estimator(ghg_model, data.loc[:, var_col], [i], random_state=10, ax=ax)
        ax.set_title(f'Partial Dependence of {var_col[i]}')
        pdf.savefig(fig)
        plt.close(fig)
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

def plotY():
    plt.scatter(y_test, y_test_pred, color='g')
    plt.plot(y_test, y_pred, color='k', label='Regression line')
    plt.plot(y_test, y_test, linestyle='dotted', color='gray', label='1:1 line')
    plt.xlabel('Actual ' + y_field)
    plt.ylabel("Predicted " + y_field)
    plt.title("Test set (y_test)")
    # Make axes of equal extents
    axes_lim = np.ceil(max(max(y_test), max(y_test_pred))) + 2
    plt.xlim((0, axes_lim))
    plt.ylim((0, axes_lim))
    plt.legend()

plotFeatureImportance()
plotY()
