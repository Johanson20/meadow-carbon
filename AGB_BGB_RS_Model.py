# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""

import os
import ee
import pickle
import warnings
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
warnings.filterwarnings("ignore")


def calculateIndices(image):
    # rename bands and  normalize raw reflectance values
    image = image.rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'QA'])
    scaled_bands = image.select(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2']).multiply(2.75e-05).add(-0.2)
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
    # mask out cloud based on bits in QA_pixel
    qa = image.select('QA')
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = dilated_cloud.And(cirrus).And(cloud).And(cloudShadow)
    return image.updateMask(cloud_mask)


# This is used to extract all band values and relevant attributes    
def extract_band_values(image):
    values = image.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=30)
    date = image.date().format('YYYY-MM-dd')
    driver = image.get('SPACECRAFT_ID')
    utm_zone = image.get('UTM_ZONE')
    return ee.Feature(None, values).set('Date', date).set('Driver', driver).set('UTM', utm_zone)


# get band values at peak EVI date, number of wet and snow days, as well as integrals over growing season
def getPeakBandValues(point, year, sortAscending=False):
    spatial_filtered = landsat[year].filterBounds(point).map(maskCloud)
    
    # extract all band values and drop NAs due to cloud masking
    band_values = spatial_filtered.map(extract_band_values).getInfo()['features']
    if not band_values:
        return [{'Blue': None}, {'Blue': None}, 0, 0]
    values = []
    for feature in band_values:
        values.append(feature['properties'])
    df = pd.DataFrame(values)
    df['Date'] = pd.to_datetime(df['Date'])
    # drop duplicate dates (order of preservation is landsat 9, then 8, then 7)
    df.drop_duplicates(subset='Date', inplace=True)
    df.dropna(inplace=True)
    
    # Set the 'date' column as the index, re-index and linearly interpolate bands at daily frequency
    df_daily = df.set_index('Date')
    date_range = pd.date_range(start=str(int(year)-1)+"-10-01", end=year+"-10-01", freq='D')[:-1]
    df_daily = df_daily.reindex(date_range).interpolate(method='linear').ffill().bfill().drop(['Driver', 'UTM'], axis=1)
    
    # sort by EVI to get band values at peak EVI date
    # compute integrals for growing season (when NDVI >= 0.2, NDWI <= 0.5 and NDSI <= 0.2)
    df['Date'] = df['Date'].dt.strftime("%Y-%m-%d")
    df.loc[df['NDSI'] > 0.2, 'EVI'] = 0
    band_values = df.sort_values('EVI', ascending=False, ignore_index=True).loc[0,:]
    df_daily.dropna(inplace=True)
    integrals = df_daily[df_daily.NDSI <= 0.2]
    # compute number of snow days (NDSI > 0.2) and number of wet days (NDWI > 0.5)
    no_snow_days = len(date_range) - integrals.shape[0]
    no_wet_days = integrals[integrals.NDWI > 0.5].shape[0]
    integrals = integrals[(integrals.NDWI <= 0.5) & (integrals.NDVI >= 0.2)]
    integrals = integrals.sum()

    return [band_values, integrals, no_snow_days, no_wet_days]


def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=30)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=30)


# reads Landsat data, flow accumulation, daymet, terraclimate and DEM data (for slope and elevation)
landsat9_collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat7_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
# flow accumulation (463.83m resolution); slope and elevation (10.2m resolution); 
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1').resample('bilinear').reproject(crs="EPSG:32610", scale=30)
dem = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32610", scale=30)
slopeDem = ee.Terrain.slope(dem)
daymet = ee.ImageCollection("NASA/ORNL/DAYMET_V4").select('swe').map(resample10)
terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(['def', 'aet', 'pr']).map(resample10)
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select(['tmmn', 'tmmx']).map(resample10)

flow_acc_11 = ee.Image("WWF/HydroSHEDS/15ACC").select('b1').resample('bilinear').reproject(crs="EPSG:32611", scale=30)
dem_11 = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32611", scale=30)
slopeDem_11 = ee.Terrain.slope(dem_11)
daymet_11 = ee.ImageCollection("NASA/ORNL/DAYMET_V4").select('swe').map(resample11)
terraclimate_11 = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(['def', 'aet', 'pr']).map(resample11)
gridmet_11 = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select(['tmmn', 'tmmx']).map(resample11)

# merge landsat, then extract unique years and create a dictionary of landsat data for each year
landsat_collection = landsat9_collection.merge(landsat8_collection).merge(landsat7_collection).map(calculateIndices)
years = set([x[:4] for x in data['SampleDate']])
landsat = {}
for year in years:
    landsat[year] = landsat_collection.filterDate(str(int(year)-1)+"-10-01", year+"-10-01")

Blue, Green, Red, NIR, SWIR_1, SWIR_2 = [], [], [], [], [], []
NDVI, NDWI, EVI, SAVI, BSI, NDSI, NDPI = [], [], [], [], [], [], []
flow, slope, elevation, wet, snowy = [], [], [], [], []
mean_annual_pr, swe, et, cdef, peak_dates, driver  = [], [], [], [], [], []
dBlue, dGreen, dRed, dNIR, dSWIR_1, dSWIR_2 = [], [], [], [], [], []
dNDVI, dNDWI, dEVI, dSAVI, dBSI, dNDSI, dNDPI = [], [], [], [], [], [], []
min_temp, max_temp = [], []

# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    year, month, day = target_date.split("-")
    next_month = str(int(month)+1) if int(month) > 8 else "0" + str(int(month)%12+1)
    if int(year) > 2023:    # 2024 data still seems unavailable
        data.drop(idx, inplace=True)
        print("Row", idx, "dropped!")
        continue
    
    band_values, integrals, snow_days, wet_days = getPeakBandValues(point, year)
    
    if not band_values['Blue']:     # drop rows that returned no value
        data.drop(idx, inplace=True)
        print("Row", idx, "dropped!")
        continue
    
    if not integrals['Blue']:
        for band in integrals.index:
            integrals[band] = 0
    
    # compute values from daymetv4 (1km resolution) and gridmet/terraclimate (resolution of both is 4,638.3m)
    mycrs = 'EPSG:326' + str(band_values['UTM'])
    if mycrs == "EPSG:32611":
        tclimate = terraclimate_11.filterBounds(point).filterDate(str(int(year)-1)+"-10-01", year+"-10-01").sum()
        daymetv4 = daymet_11.filterBounds(point).filterDate(year + '-04-01', year + '-04-02').first()
        max_tvalues = gridmet_11.filterBounds(point).filterDate(year+"-"+month+"-01", year+"-"+next_month+"-01").max()
        min_tvalues = gridmet_11.filterBounds(point).filterDate(year+"-"+month+"-01", year+"-"+next_month+"-01").min()
        elev = dem_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
        slope_value = slopeDem_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
        flow_value = flow_acc_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    else:
        tclimate = terraclimate.filterBounds(point).filterDate(str(int(year)-1)+"-10-01", year+"-10-01").sum()
        daymetv4 = daymet.filterBounds(point).filterDate(year + '-04-01', year + '-04-02').first()
        max_tvalues = gridmet.filterBounds(point).filterDate(year+"-"+month+"-01", year+"-"+next_month+"-01").max()
        min_tvalues = gridmet.filterBounds(point).filterDate(year+"-"+month+"-01", year+"-"+next_month+"-01").min()
        elev = dem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['elevation']
        slope_value = slopeDem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['slope']
        flow_value = flow_acc.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        
    swe_value = daymetv4.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['swe']
    tclimate = tclimate.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    mean_pr = tclimate['pr']
    cdef_value = tclimate['def']
    aet = tclimate['aet']
    temps = min_tvalues.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    tmin = temps['tmmn']
    temps = max_tvalues.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    tmax = temps['tmmx']
    
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
    
    dBlue.append(integrals['Blue'])
    dGreen.append(integrals['Green'])
    dRed.append(integrals['Red'])
    dNIR.append(integrals['NIR'])
    dSWIR_1.append(integrals['SWIR_1'])
    dSWIR_2.append(integrals['SWIR_2'])
    dNDVI.append(integrals['NDVI'])
    dNDWI.append(integrals['NDWI'])
    dEVI.append(integrals['EVI'])
    dSAVI.append(integrals['SAVI'])
    dBSI.append(integrals['BSI'])
    dNDSI.append(integrals['NDSI'])
    dNDPI.append(integrals['NDPI'])
    
    wet.append(wet_days)
    snowy.append(snow_days)
    mean_annual_pr.append(mean_pr)
    flow.append(flow_value)
    elevation.append(elev)
    slope.append(slope_value)
    swe.append(swe_value)
    et.append(aet)
    cdef.append(cdef_value)
    peak_dates.append(band_values['Date'])
    driver.append(band_values['Driver'])
    min_temp.append(tmin)
    max_temp.append(tmax)
    
    if idx%50 == 0: print(idx, end=' ')

# checks if they are all cloud free (should equal data.shape[0])
len([x for x in NIR if x])

data['Blue'] = Blue
data['Green'] = Green
data['Red'] = Red
data['NIR'] = NIR
data['SWIR_1'] = SWIR_1
data['SWIR_2'] = SWIR_2
data['peak_date'] = peak_dates
data['Driver'] = driver

data['Cdef'] = cdef
data['Elevation'] = elevation
data['AET'] = et
data['Flow'] = flow
data['Slope'] = slope
data['SWE'] = swe
data['Annual_Precipitation'] = mean_annual_pr
data['Snow_days'] = snowy
data['Wet_days'] = wet
data['Minimum_temperature'] = min_temp
data['Maximum_temperature'] = max_temp

data['NIR_Green'] = data['NIR']/data['Green']
data['NIR_Red'] = data['NIR']/data['Red']
data['NDVI'] = NDVI
data['NDWI'] = NDWI
data['EVI'] = EVI
data['SAVI'] = SAVI
data['BSI'] = BSI
data['NDSI'] = NDSI
data['NDPI'] = NDPI

data['dBlue'] = dBlue
data['dGreen'] = dGreen
data['dRed'] = dRed
data['dNIR'] = dNIR
data['dSWIR_1'] = dSWIR_1
data['dSWIR_2'] = dSWIR_2
data['dNDVI'] = dNDVI
data['dNDWI'] = dNDWI
data['dEVI'] = dEVI
data['dSAVI'] = dSAVI
data['dBSI'] = dBSI
data['dNDSI'] = dNDSI
data['dNDPI'] = dNDPI

data.reset_index(drop=True, inplace=True)
data.head()

# write updated dataframe to new csv file
data.to_csv(filename.split(".csv")[0] + "_Data.csv", index=False)


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
data = pd.read_csv("csv/Belowground Biomass_RS Model_Data.csv")
# data = pd.read_csv("csv/BGB_summarized_soil_depths.csv")  # soil carbon with summarized depths
# data = pd.read_csv("csv/BGB_separated_soil_depths.csv")   # soil carbon with separated depths
data.head()
# data['SampleDate'] = pd.to_datetime(data['SampleDate'])
# data = data[data['SampleDate'].dt.year.isin([2015, 2016])]
# confirm column names first
cols = data.columns
# cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)
# data['ID'].value_counts()      # number of times same ID was sampled

# remove irrelevant columns for ML and determine X and Y variables
var_col =  list(cols[20:26]) + list(cols[-13:])
# var_col =  list(cols[20:26]) + list(cols[-18:])   # soil carbon with summarized depths
# var_col =  list(cols[20:26]) + list(cols[-29:])   # soil carbon with separated depths
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

bgb_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, n_estimators=75, subsample=0.8, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
'''# soil carbon with summarized depths
bgb_model = GradientBoostingRegressor(learning_rate=0.07, max_depth=3, n_estimators=200, subsample=0.3, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
# soil carbon with separated depths
bgb_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=6, n_estimators=75, subsample=0.8, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)'''
bgb_84_model = GradientBoostingRegressor(loss="quantile", learning_rate=0.1, alpha=0.8413, max_depth=4, 
                                      n_estimators=75, subsample=0.8, validation_fraction=0.2, n_iter_no_change=50,  
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
var_col =  list(cols[18:24]) + list(cols[-13:])
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

agb_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, n_estimators=25, subsample=0.8, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
agb_84_model = GradientBoostingRegressor(loss="quantile", learning_rate=0.1, alpha=0.8413, max_depth=5, 
                                      n_estimators=25, subsample=0.8, validation_fraction=0.2, n_iter_no_change=50,  
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