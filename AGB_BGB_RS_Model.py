# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""


import os
import ee
import pickle
import pandas as pd

mydir = "Code"      # adjust directory
os.chdir(mydir)

# read csv file and convert dates from strings to datetime
filename = "csv/Belowground Biomass_RS Model.csv"
# REPEAT same for AGB
# filename = "csv/Aboveground Biomass_RS Model.csv"
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


def calculateIndices(image):
    # rename bands and  normalize raw reflectance values
    scaled_bands = image.select(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2']).multiply(1e-4)
    image = image.addBands(scaled_bands, overwrite=True)
    
    # add indices
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndwi = image.normalizedDifference(['NIR', 'SWIR_1']).rename('NDWI')
    ndsi = image.normalizedDifference(['Green', 'SWIR_1']).rename('NDSI')
    evi = image.expression("2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'BLUE': image.select('Blue')}).rename('EVI')
    savi = image.expression("1.5 * ((NIR - RED) / (NIR + RED + 0.5))", {'NIR': image.select('NIR'), 'RED': image.select('Red')}).rename('SAVI')
    bsi = image.expression("((RED + SWIR_1) - (NIR + BLUE)) / (RED + SWIR_1 + NIR + BLUE)", {'RED': image.select('Red'), 'SWIR_1': image.select('SWIR_1'), 'NIR': image.select('NIR'), 'BLUE': image.select('Blue')}).rename('BSI')
    ndpi = image.expression("(NIR - ((0.56 * RED) + (0.44 * SWIR_2))) / (NIR + ((0.56 * RED) + (0.44 * SWIR_2)))", {'NIR': image.select('NIR'), 'RED': image.select('Red'), 'SWIR_2': image.select('SWIR_2')}).rename('NDPI')
    
    return image.addBands([ndvi, ndwi, evi, savi, bsi, ndsi, ndpi])


def maskCloud(image):
    image = image.rename(['Blue','Green','Red','RE_1','RE_2','RE_3','NIR','RE_4','SWIR_1','SWIR_2','SCL'])
    # mask out cloud based on bits in SCL band
    scl = image.select('SCL')
    cloud_mask = scl.neq(3).And(scl.lt(8))
    return image.updateMask(cloud_mask)


# Calculates absolute time difference (in days) from a target date, in which the images are acquired
def calculate_time_difference(image):
    time_difference = ee.Number(image.date().difference(target_date, 'day')).abs()
    return image.set('time_difference', time_difference)


def getBandValues(img_collection, point, target_date, bufferDays = 15):
    # filter landsat images by location and dates about a 15-day radius and sort by proximity to sample date
    spatial_filtered = img_collection.filterBounds(point)
    temporal_filtered = spatial_filtered.filterDate(ee.Date(target_date).advance(-bufferDays, 'day'), ee.Date(target_date).advance(bufferDays, 'day'))
    # Map the ImageCollection over time difference and sort to get image of closest date
    sorted_collection = temporal_filtered.map(calculate_time_difference).sort('time_difference')
    noImages = sorted_collection.size().getInfo()
    
    if not noImages:
        return [{"Blue": None}, None]
    
    image_list = sorted_collection.toList(sorted_collection.size())
    nImage, band_values = 0, {'Blue': None}
    
    # repeatedly check for cloud free pixels (non-null value) in landsat 8, or checks in landsat 7
    while band_values['Blue'] == None and nImage < noImages:
        nearest_image = ee.Image(image_list.get(nImage))
        nImage += 1
        value = nearest_image.getInfo()['properties']
        band_values = nearest_image.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    
    return [band_values, value['time_difference']]


def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=10)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=10)


sentinel1 = ee.ImageCollection("COPERNICUS/S1_GRD").select(['VH', 'VV'])
sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','SCL'])
sentinel2_10 = sentinel2.map(resample10).map(maskCloud).map(calculateIndices)
sentinel2_11 = sentinel2.map(resample11).map(maskCloud).map(calculateIndices)

target_date = ''
Blue, Green, Red, NIR, SWIR_1, SWIR_2 = [], [], [], [], [], []
NDVI, NDWI, EVI, SAVI, BSI, NDSI, NDPI = [], [], [], [], [], [], []
RE_1, RE_2, RE_3, RE_4, VV, VH, CRS, time_diff = [], [], [], [], [], [], [], []

# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    
    if x <= -120:   # zone 11 longitudes are (126W, 120W)
        mycrs = "EPSG:32611"
        band_values, t_diff = getBandValues(sentinel2_11, point, target_date)
    else:
        mycrs = "EPSG:32610"
        band_values, t_diff = getBandValues(sentinel2_10, point, target_date)
    if not band_values['Blue']:
        data.drop(idx, inplace=True)
        print("Row", idx, "dropped!")
        continue
    
    sent = sentinel1.filterBounds(point).filterDate(ee.Date(target_date).advance(-3, 'day'), ee.Date(target_date).advance(3, 'day'))
    try:
        sent_values = sent.first().reduceRegion(ee.Reducer.mean(), point, 10).getInfo()
    except:
        sent_values = {"VH": None, "VV": None}
    
    Blue.append(band_values['Blue'])
    Green.append(band_values['Green'])
    Red.append(band_values['Red'])
    NIR.append(band_values['NIR'])
    SWIR_1.append(band_values['SWIR_1'])
    SWIR_2.append(band_values['SWIR_2'])
    NDVI.append(band_values['NDVI'])
    NDWI.append(band_values['NDWI'])
    EVI.append(band_values['EVI'])
    SAVI.append(band_values['SAVI'])
    BSI.append(band_values['BSI'])
    NDSI.append(band_values['NDSI'])
    NDPI.append(band_values['NDPI'])
    
    RE_1.append(band_values['RE_1'])
    RE_2.append(band_values['RE_2'])
    RE_3.append(band_values['RE_3'])
    RE_4.append(band_values['RE_4'])
    VV.append(sent_values['VV'])
    VH.append(sent_values['VH'])
    time_diff.append(t_diff)
    CRS.append(mycrs)
    
    if idx%50 == 0: print(idx, end=' ')


data['Blue'] = Blue
data['Green'] = Green
data['Red'] = Red
data['NIR'] = NIR
data['SWIR_1'] = SWIR_1
data['SWIR_2'] = SWIR_2
data['NDVI'] = NDVI
data['NDWI'] = NDWI
data['EVI'] = EVI
data['SAVI'] = SAVI
data['BSI'] = BSI
data['NDSI'] = NDSI
data['NDPI'] = NDPI

data['RE_1'] = RE_1
data['RE_2'] = RE_2
data['RE_3'] = RE_3
data['RE_4'] = RE_4
data['VH'] = VH
data['VV'] = VV
data['CRS_Zone'] = CRS
data['Days_of_data_acquisition_offset'] = time_diff

data.to_csv("files/Belowground Biomass_RS Model_Data.csv", index=False)  

# ML training starts here
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read csv containing random samples
data = pd.read_csv("files/Belowground Biomass_RS Model_Data.csv")
data.head()
# confirm column names first
cols = data.columns
# cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)
# data['ID'].value_counts()      # number of times same ID was sampled

# remove irrelevant columns for ML and determine X and Y variables
var_col =  list(cols[27:-2]) + ['dNDPI']
y_field = 'Roots.kg.m2'
# subdata excludes other measured values which can be largely missing (as we need to assess just one output at a time)
subdata = data.loc[:, ([y_field] + var_col)]

# check for missing/null values across columns and rows respectively (confirm results below should be all 0)
sum(subdata.isnull().any(axis=0) == True)
sum(subdata[y_field].isnull())

# drop values with root biomass greater than 12 (outliers)
# outlierIds = data[data[y_field] > 12].index
# data.drop(outlierIds, inplace = True)
# if NAs where found (results above are not 0) in one of them (e.g. Y)
nullIds = list(np.where(subdata[y_field].isnull())[0])    # null IDs
data.drop(nullIds, inplace = True)
data.dropna(subset=[y_field], inplace=True)
data.dropna(subset=var_col, inplace=True)
data.reset_index(drop=True, inplace=True)
# make scatter plots of relevant variables from raw dataframe
with PdfPages('files/BGB_Scatter_plots.pdf') as pdf:
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

bgb_model = GradientBoostingRegressor(learning_rate=0.16, max_depth=14, n_estimators=25, subsample=0.8, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
bgb_84_model = GradientBoostingRegressor(loss="quantile", learning_rate=0.16, alpha=0.8413, max_depth=6, 
                                      n_estimators=50, subsample=0.5, validation_fraction=0.2, n_iter_no_change=50,  
                                      max_features='log2', random_state=10)

bgb_model.fit(X_train, y_train)
bgb_84_model.fit(X_train, y_train)
# Make partial dependence plots
with PdfPages('files/BGB_partial_dependence_plots.pdf') as pdf:
    for i in range(len(var_col)):
        fig, ax = plt.subplots(figsize=(8, 6))
        PartialDependenceDisplay.from_estimator(bgb_model, data.loc[:, var_col], [i], random_state=10, ax=ax)
        ax.set_title(f'Partial Dependence of {var_col[i]}')
        pdf.savefig(fig)
        plt.close(fig)
with PdfPages('files/BGB_1_1_plot.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(8, 6))
    y_test_pred = bgb_model.predict(X_test)
    y_test_84_pred = bgb_84_model.predict(X_test)
    sns.regplot(y=y_test_pred, x=y_test_84_pred, line_kws={"color":"blue"}, ax=ax, label="84th quantile prediction")
    sns.regplot(y=y_test_pred, x=y_test_pred, line_kws={"color":"red"}, ax=ax, label=f"Mean prediction: R = {round(np.corrcoef(y_test_84_pred, y_test_pred)[1][0], 5)}")
    ax.set_title('Scatter plot of 84th_quantile vs mean')
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)
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


# same procedure as above
data = pd.read_csv("csv/Aboveground Biomass_RS Model_Data.csv")
data.head()
# confirm column names first
cols = data.columns
# cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)
# data['ID'].value_counts()   # number of times same ID was sampled

# remove irrelevant columns for ML and determine X and Y variables
var_col = list(cols[25:-2]) + ['dNDPI']
y_field = 'HerbBio.g.m2'
# subdata excludes other measured values which can be largely missing (as we need to assess just one output at a time)
subdata = data.loc[:, ([y_field] + var_col)]

# check for missing/null values across columns and rows respectively (confirm results below should be all 0)
sum(subdata.isnull().any(axis=0) == True)
sum(subdata[y_field].isnull())

# if NAs where found (results above are not 0) in one of them (e.g. Y)
nullIds = list(np.where(subdata[y_field].isnull())[0])    # null IDs
data.drop(nullIds, inplace = True)
data.dropna(subset=[y_field], inplace=True)
data.dropna(subset=var_col, inplace=True)
data.reset_index(drop=True, inplace=True)
# make scatter plots of relevant variables from raw dataframe
with PdfPages('files/AGB_Scatter_plots.pdf') as pdf:
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

agb_model = GradientBoostingRegressor(learning_rate=0.07, max_depth=6, n_estimators=50, subsample=0.7, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
agb_84_model = GradientBoostingRegressor(loss="quantile", learning_rate=0.07, alpha=0.8413, max_depth=6, 
                                      n_estimators=50, subsample=0.7, validation_fraction=0.2, n_iter_no_change=50,  
                                      max_features='log2', random_state=10)
agb_model.fit(X_train, y_train)
agb_84_model.fit(X_train, y_train)
# Make partial dependence plots
with PdfPages('files/AGB_partial_dependence_plots.pdf') as pdf:
    for i in range(len(var_col)):
        fig, ax = plt.subplots(figsize=(8, 6))
        PartialDependenceDisplay.from_estimator(agb_model, data.loc[:, var_col], [i], random_state=10, ax=ax)
        ax.set_title(f'Partial Dependence of {var_col[i]}')
        pdf.savefig(fig)
        plt.close(fig)
with PdfPages('files/AGB_1_1_plot.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(8, 6))
    y_test_pred = agb_model.predict(X_test)
    y_test_84_pred = agb_84_model.predict(X_test)
    sns.regplot(y=y_test_pred, x=y_test_84_pred, line_kws={"color":"blue"}, ax=ax, label="84th quantile prediction")
    sns.regplot(y=y_test_pred, x=y_test_pred, line_kws={"color":"red"}, ax=ax, label=f"Mean prediction: R = {round(np.corrcoef(y_test_84_pred, y_test_pred)[1][0], 5)}")
    ax.set_title('Scatter plot of 84th_quantile vs mean')
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)
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
plotY()


f = open('csv/models.pckl', 'wb')
pickle.dump([ghg_model, agb_model, bgb_model], f)
f.close()
f = open('csv/sd_models.pckl', 'wb')
pickle.dump([ghg_84_model, agb_84_model, bgb_84_model], f)
f.close()


'''
# GHG Results:
{'RMSE': [1.38516, 0.03, 250, 0.8, 11, 0.2]}

# BGB Results:
{'RMSE': [1.37666, 0.16, 50, 0.9, 12, 0.2]}

# AGB Results: 
{'RMSE': [109.28577, 0.33, 75, 0.9, 18, 0.2]}
'''