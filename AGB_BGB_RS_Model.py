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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import randint, uniform
import numpy as np

# read csv containing random samples
data = pd.read_csv("Belowground Biomass_RS Model_Data.csv")
# data = pd.read_csv("Aboveground Biomass_RS Model_Data.csv")
data.head()
cols = data.columns
# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in cols[6:] if c not in ['percentC', '...1', 'peak_date']]
# var_col = [c for c in cols[5:] if c != 'peak_date']
X = data.loc[:, var_col[3:]]
# X = data.loc[:, var_col[2:]]
Y = data.loc[:, 'Roots.kg.m2']
# Y = data.loc[:, 'HerbBio.g.m2']
sum(X['Blue'].isna())  # check for missing/null values
sum(X.isnull().any(axis=1) == True)   # set axis=0 for missing column values (1 for rows)

# split X and Y into training (80%) and test data (20%), random state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

''' Before using gradient boosting, optimize hyperparameters either:
    by randomnly selecting from a range of values using RandomizedSearchCV,
    or by grid search which searches through all provided possibilities with GridSearchCV
'''
# optimize hyperparameters with RandomizedSearchCV on the training data (takes 30ish minutes)
parameters = {'learning_rate': uniform(), 'subsample': uniform(), 
              'n_estimators': randint(5, 5000), 'max_depth': randint(2, 10)}
randm = RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=parameters,
                           cv=10, n_iter=50, n_jobs=1)
randm.fit(X_train, y_train)
randm.best_estimator_
randm.best_params_      # outputs all parameters of ideal estimator

# same process above but with GridSearchCV for comparison (takes even longer)
parameters = {'learning_rate': [0.01, 0.03, 0.05], 'subsample': [0.4, 0.5, 0.6], 
              'n_estimators': [1000, 2500, 5000], 'max_depth': [6,7,8]}
grid = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=parameters, cv=10, n_jobs=1)
grid.fit(X_train, y_train)
grid.best_params_