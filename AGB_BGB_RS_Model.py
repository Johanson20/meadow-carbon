# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""

import os
import ee
import pickle
import warnings
import numpy as np
import pandas as pd

mydir = "Code"      # adjust directory
os.chdir(mydir)

# read csv file and convert dates from strings to datetime
filename = "csv/Belowground Biomass_RS Model.csv"
# REPEAT same for AGB
# filename = "csv/Aboveground Biomass_RS Model.csv"
data = pd.read_csv(filename)
data.head()
'''
# fix coordinate values that are often mistyped
data.columns
data.loc[:, ['Longitude', 'Latitude']].describe()
idx = data[data['Longitude'] > -116].index
data.loc[idx, 'Longitude'] -= 100
data.drop("Unnamed: 0", axis=1, inplace=True)
data.to_csv(filename, index=False)
'''
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
    # normalize raw reflectance values and calculate indices
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
    # rename bands and mask out cloud based on bits in QA_pixel
    image = image.rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'QA'])
    qa = image.select('QA')
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = dilated_cloud.And(cirrus).And(cloud).And(cloudShadow)
    return image.updateMask(cloud_mask)


def getBandValues(image):
    # extract band values (with indices) alongside UTM, date and timezone of landsat image
    image = calculateIndices(image)
    values = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30)
    date = image.date().format('YYYY-MM-dd')
    utm_zone = image.get('UTM_ZONE')
    return ee.Feature(None, values).set('Date', date).set('UTM', utm_zone)


# get band values at peak EVI date, number of wet and snow days, as well as integrals over growing season
def extractAllValues(landsat, year):
    # get band values and ensure it is not null
    landsat_values = landsat.map(getBandValues).getInfo()['features']
    if not landsat_values:
        return [{'Blue': None}, {'Blue': None}, 0, {'Blue': None}, 0, 0]
    values = []
    # convert band values and other properties to a dataframe for manipulation
    for feature in landsat_values:
        values.append(feature['properties'])
    df = pd.DataFrame(values)
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop_duplicates(subset='Date', inplace=True)
    df.dropna(inplace=True)
    
    # Set the 'date' column as the index, re-index and linearly interpolate bands at daily frequency
    df_daily = df.set_index('Date')
    years = sorted(df_daily.index.year.unique())
    date_range = pd.date_range(start=str(years[0])+"-10-01", end=str(years[-1])+"-10-01", freq='D')[:-1]
    df_daily = df_daily.reindex(date_range).interpolate(method='linear').ffill().bfill()
    df_daily.dropna(inplace=True)
    # also extract two sets of values: the current year and the previous 5 years.
    df_prev_5 = df_daily.loc[:str(years[-1]-1)+"-09-30"]
    df_year = df_daily.loc[str(years[-1]-1)+"-10-01":]
    
    # compute number of snow days (NDSI > 0.2) and number of wet days (NDWI > 0.5)
    integrals = df_year[df_year.NDSI <= 0.2]
    no_snow_days = len(df_year) - len(integrals)
    no_wet_days = integrals[integrals.NDWI > 0.5].shape[0]
    # Active growing season is also when NDVI >= 0.2 without snow or water coverage
    integrals = integrals[(integrals.NDWI <= 0.5) & (integrals.NDVI >= 0.2)]
    integrals = integrals.sum()
    # Compute indices for previous 5 years of june and sept
    june_prev_5 = df_prev_5[df_prev_5.index.month == 6]
    sept_prev_5 = df_prev_5[df_prev_5.index.month == 9]
    return [june_prev_5.mean(), sept_prev_5.mean(), df['UTM'].mode().iloc[0], integrals, no_snow_days, no_wet_days]


# resample images collections whose band values are not 30m for both UTM zones
def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=30)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=30)


# reads and merge Landsat data, and other datasets
landsat9_collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat7_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsat_collection = landsat9_collection.merge(landsat8_collection).merge(landsat7_collection).map(maskCloud)

# flow accumulation (463.83m resolution); slope and elevation (10.2m resolution); gridmet/terraclimate (4,638.3m resolution)
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1').resample('bilinear').reproject(crs="EPSG:32610", scale=30)
dem = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32610", scale=30)
slopeDem = ee.Terrain.slope(dem)
terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(['def', 'aet', 'pr', 'swe']).map(resample10)
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select(['tmmn', 'tmmx', 'srad']).map(resample10)

flow_acc_11 = ee.Image("WWF/HydroSHEDS/15ACC").select('b1').resample('bilinear').reproject(crs="EPSG:32611", scale=30)
dem_11 = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32611", scale=30)
slopeDem_11 = ee.Terrain.slope(dem_11)
terraclimate_11 = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(['def', 'aet', 'pr', 'swe']).map(resample11)
gridmet_11 = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select(['tmmn', 'tmmx', 'srad']).map(resample11)

# these polaris soil datasets have 30m spatial resolution (same as landsat above); lithology is 90m resolution
perc_clay = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample10)
hydra_cond = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample10)
perc_sand = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').select("b1").map(resample10)
lithology = ee.Image("CSP/ERGo/1_0/US/lithology").select("b1").resample("bilinear").reproject(crs="EPSG:32610", scale=30)
organic_m = ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').select("b1").map(resample10)
shallow_perc_clay = ee.ImageCollection(perc_clay.toList(3)).mean()
deep_perc_clay = ee.Image(perc_clay.toList(6).get(3))
shallow_hydra_cond = ee.ImageCollection(hydra_cond.toList(3)).mean()
deep_hydra_cond = ee.Image(hydra_cond.toList(6).get(3))
shallow_perc_sand = ee.ImageCollection(perc_sand.toList(3)).mean()
deep_perc_sand = ee.Image(perc_sand.toList(6).get(3))
shallow_organic_m = ee.ImageCollection(organic_m.toList(3)).mean()

# same as above but for EPSG 32611 (above is 32610)
perc_clay_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample11)
hydra_cond_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample11)
perc_sand_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').select("b1").map(resample11)
lithology_11 = ee.Image("CSP/ERGo/1_0/US/lithology").select("b1").resample("bilinear").reproject(crs="EPSG:32611", scale=30)
organic_m_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').select("b1").map(resample11)
shallow_perc_clay_11 = ee.ImageCollection(perc_clay_11.toList(3)).mean()
deep_perc_clay_11 = ee.Image(perc_clay_11.toList(6).get(3))
shallow_hydra_cond_11 = ee.ImageCollection(hydra_cond_11.toList(3)).mean()
deep_hydra_cond_11 = ee.Image(hydra_cond_11.toList(6).get(3))
shallow_perc_sand_11 = ee.ImageCollection(perc_sand_11.toList(3)).mean()
deep_perc_sand_11 = ee.Image(perc_sand_11.toList(6).get(3))
shallow_organic_m_11 = ee.ImageCollection(organic_m_11.toList(3)).mean()

dBlue, dGreen, dRed, dNIR, dSWIR_1, dSWIR_2 = [], [], [], [], [], []
dNDVI, dNDWI, dEVI, dSAVI, dBSI, dNDPI, dNDSI = [], [], [], [], [], [], []
NDWI_Summer, EVI_Summer, SAVI_Summer, BSI_Summer, NDPI_Summer = [], [], [], [], []
NDWI_Fall, EVI_Fall, SAVI_Fall, BSI_Fall, NDPI_Fall = [], [], [], [], []
flow, slope, elevation, wet, snowy, S_Rad = [], [], [], [], [], []
mean_annual_pr, swe, et, cdef, min_temp, max_temp, Organic_Matter = [], [], [], [], [], [], []
Shallow_Clay, Shallow_Hydra, Shallow_Sand, Lithology, Deep_Clay, Deep_Hydra, Deep_Sand = [], [], [], [], [], [], []

# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    year, month, day = target_date.split("-")
    # next_month = str(int(month)+1) if int(month) > 8 else "0" + str(int(month)%12+1)
    prev_5_year = str(int(year)-6) + "-10-01"
    
    landsat = landsat_collection.filterBounds(point).filterDate(prev_5_year, year+"-10-01")
    bands_June, bands_Sept, utm, integrals, snow_days, wet_days = extractAllValues(landsat, year)
    
    if not bands_June['Blue'] or not bands_Sept['Blue']:     # drop rows that returned null band values
        data.drop(idx, inplace=True)
        print("Row", idx, "dropped!")
        continue
    
    if not integrals['Blue']:   # if integrals are null (probably from null band values), assign zero to them
        for band in integrals.index:
            integrals[band] = 0
    
    # compute band values: daymetv4 (1km resolution) can also be used for SWE
    mycrs = 'EPSG:326' + str(utm)
    if mycrs == "EPSG:32611":
        tclimate = terraclimate_11.filterBounds(point).filterDate(str(int(year)-1)+"-10-01", year+"-10-01").sum()
        srad = gridmet_11.filterBounds(point).filterDate(year+"-06-01", year+"-08-31").sum()
        snow_we = terraclimate_11.filterBounds(point).filterDate(year + '-04-01', year + '-05-01').first()
        tvalues = gridmet_11.filterBounds(point).filterDate(str(int(year)-1)+"-10-01", year+"-10-01").mean()
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
        shall_org = shallow_organic_m_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    else:
        tclimate = terraclimate.filterBounds(point).filterDate(str(int(year)-1)+"-10-01", year+"-10-01").sum()
        srad = gridmet.filterBounds(point).filterDate(year+"-06-01", year+"-08-31").sum()
        snow_we = terraclimate.filterBounds(point).filterDate(year + '-04-01', year + '-05-01').first()
        tvalues = gridmet.filterBounds(point).filterDate(str(int(year)-1)+"-10-01", year+"-10-01").mean()
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
        shall_org = shallow_organic_m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        
    swe_value = snow_we.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['swe']
    tclimate = tclimate.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    mean_pr = tclimate['pr']
    cdef_value = 0.1*tclimate['def']
    aet = 0.1*tclimate['aet']
    temps = tvalues.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    rad = srad.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['srad']
    tmin = temps['tmmn']
    tmax = temps['tmmx']
    
    NDWI_Summer.append(bands_June['NDWI'])
    EVI_Summer.append(bands_June['EVI'])
    SAVI_Summer.append(bands_June['SAVI'])
    BSI_Summer.append(bands_June['BSI'])
    NDPI_Summer.append(bands_June['NDPI'])
    NDWI_Fall.append(bands_Sept['NDWI'])
    EVI_Fall.append(bands_Sept['EVI'])
    SAVI_Fall.append(bands_Sept['SAVI'])
    BSI_Fall.append(bands_Sept['BSI'])
    NDPI_Fall.append(bands_Sept['NDPI'])
    
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
    dNDPI.append(integrals['NDPI'])
    dNDSI.append(integrals['NDSI'])
    
    wet.append(wet_days)
    snowy.append(snow_days)
    mean_annual_pr.append(mean_pr)
    flow.append(flow_value)
    elevation.append(elev)
    slope.append(slope_value)
    swe.append(swe_value)
    et.append(aet)
    cdef.append(cdef_value)
    min_temp.append(tmin)
    max_temp.append(tmax)
    S_Rad.append(rad)
    
    Shallow_Clay.append(shallow_clay)
    Shallow_Sand.append(shallow_sand)
    Shallow_Hydra.append(np.power(10, shallow_hydra))
    Lithology.append(lith)
    Deep_Clay.append(deep_clay)
    Deep_Sand.append(deep_sand)
    Deep_Hydra.append(np.power(10, deep_hydra))
    Organic_Matter.append(np.power(10, shall_org))
    
    if idx%20 == 0: print(idx, end=' ')

# checks if they are all cloud free (should equal data.shape[0])
len([x for x in dNIR if x])

data['NDWI_June'] = NDWI_Summer
data['EVI_June'] = EVI_Summer
data['SAVI_June'] = SAVI_Summer
data['BSI_June'] = BSI_Summer
data['NDPI_June'] = NDPI_Summer
data['NDWI_Sept'] = NDWI_Fall
data['EVI_Sept'] = EVI_Fall
data['SAVI_Sept'] = SAVI_Fall
data['BSI_Sept'] = BSI_Fall
data['NDPI_Sept'] = NDPI_Fall

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
data['dNDPI'] = dNDPI
data['dNDSI'] = dNDSI

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
data['SRad'] = S_Rad

data['Shallow_Clay'] = Shallow_Clay
data['Deep_Clay'] = Deep_Clay
data['Shallow_Sand'] = Shallow_Sand
data['Deep_Sand'] = Deep_Sand
data['Shallow_Hydra_Conduc'] = Shallow_Hydra
data['Deep_Hydra_Conduc'] = Deep_Hydra
data['Lithology'] = Lithology
data['Organic_Matter'] = Organic_Matter

data.reset_index(drop=True, inplace=True)
data.head()

# write updated dataframe to new csv file
data.to_csv(filename.split(".csv")[0] + "_5_year_Data.csv", index=False)
# data.to_csv(filename.split(".csv")[0] + "_Data.csv", index=False)


# ML training starts here
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read csv containing random samples
data = pd.read_csv("csv/Belowground Biomass_RS Model_5_year_Data.csv")  # for the 5-year averaged data
'''data = pd.read_csv("csv/BGB_summarized_soil_depths.csv")  # soil carbon with summarized depths
data = pd.read_csv("csv/Belowground Biomass_RS Model_Data.csv")   # for the old "Data" (without 5 year averages)
data = pd.read_csv("csv/BGB_separated_soil_depths.csv")   # soil carbon with separated depths
data['SampleDate'] = pd.to_datetime(data['SampleDate'])
data = data[data['SampleDate'].dt.year.isin([2015, 2016])]
# confirm column names first
cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column'''
data.head()
cols = data.columns
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)
# data['ID'].value_counts()      # number of times same ID was sampled

# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in list(cols[11:]) if c not in ['dNDSI', 'Cdef', 'AET', 'Flow', 'Wet_days', 'Lithology', 'EVI_Sept', 'NDWI_Sept', 'Shallow_Clay', 'EVI_June', 'dNDWI', 'dEVI', 'dRed', 'NDPI_June', 'SAVI_June', 'dSWIR_2', 'dNDVI']]
'''var_col = list(cols[20:26]) + list(cols[-13:])   # for the old "Data" (without 5 year averages)
var_col = list(cols[20:26]) + list(cols[-18:])   # soil carbon with summarized depths
var_col = list(cols[20:26]) + list(cols[-29:])   # soil carbon with separated depths'''
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

# bin the dataset based on BGB values
data[y_field].describe()
data['BGB_bin_class'] = 0
for value in range(2, 16, 2):   # max rounded value is 16
    mask = (data[y_field] > value) & (data[y_field] <= (value+2))
    data.loc[mask, 'BGB_bin_class'] = value//2
data['BGB_bin_class'].describe()

# split data into training (80%) and test data (20%) by IDs, random state ensures reproducibility
train_df = data.groupby('BGB_bin_class', group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=10))
test_data = data[~data.index.isin(train_df.index)]
# upsample the training dataset so that all bins have same amount of rows
max_size = train_df['BGB_bin_class'].value_counts().max()
train_data = (train_df.groupby('BGB_bin_class', group_keys=False)
    .apply(lambda x: resample(x, replace=True, n_samples=max_size, random_state=10)).reset_index(drop=True))
train_data['BGB_bin_class'].value_counts()

X_train, y_train = train_data.loc[:, var_col], train_data[y_field]
X_test, y_test = test_data.loc[:, var_col], test_data[y_field]

# for the 5-year averaged data
bgb_model = GradientBoostingRegressor(learning_rate=0.2, max_depth=9, n_estimators=75, subsample=1.0, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
'''# soil carbon with summarized depths
bgb_model = GradientBoostingRegressor(learning_rate=0.07, max_depth=3, n_estimators=200, subsample=0.3, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
# for the old "Data" (without 5 year averages)
bgb_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, n_estimators=75, subsample=0.8, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
# soil carbon with separated depths
bgb_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=6, n_estimators=75, subsample=0.8, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)'''
bgb_84_model = GradientBoostingRegressor(loss="quantile", alpha=0.8413, learning_rate=0.3, max_depth=4, n_estimators=50,
                                         subsample=0.7, validation_fraction=0.2, n_iter_no_change=50, max_features='log2',
                                         verbose=1, random_state=10)

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
feat_imp = bgb_model.feature_importances_
sorted_idx = np.argsort(feat_imp)
pos = np.arange(sorted_idx.shape[0]) + 0.5

def plotFeatureImportance():
    plt.barh(pos, feat_imp[sorted_idx], align="center")
    plt.yticks(pos, np.array(bgb_model.feature_names_in_)[sorted_idx])
    plt.title("Feature Importance")

def plotTestY():
    # Make regression line over y_test and it's predictions
    regressor = LinearRegression()
    test_y = np.array(y_test).reshape(-1,1)
    test_pred_y = np.array(y_test_pred).reshape(-1,1)
    regressor.fit(test_y, test_pred_y)
    y_pred = regressor.predict(test_y)
    plt.scatter(test_y, y_test_pred, color='g')
    plt.plot(test_y, y_pred, color='k', label='Regression line')
    plt.plot(test_y, test_y, linestyle='dotted', color='gray', label='1:1 line')
    plt.xlabel('Actual ' + y_field)
    plt.ylabel("Predicted " + y_field)
    plt.title(f"Test set (y_test); R = {np.round(test_corr[0][1], 4)}")
    # Make axes of equal extents
    axes_lim = np.ceil(max(max(test_y), max(test_pred_y))) + 2
    plt.xlim((0, axes_lim))
    plt.ylim((0, axes_lim))
    plt.legend()

def plotTrainY():
    # Make regression line over y_train and it's predictions
    regressor = LinearRegression()
    train_y = np.array(y_train).reshape(-1,1)
    train_pred_y = np.array(y_train_pred).reshape(-1,1)
    regressor.fit(train_y, train_pred_y)
    y_pred = regressor.predict(train_y)
    plt.scatter(train_y, train_pred_y, color='g')
    plt.plot(train_y, y_pred, color='k', label='Regression line')
    plt.plot(train_y, train_y, linestyle='dotted', color='gray', label='1:1 line')
    plt.xlabel('Actual ' + y_field)
    plt.ylabel("Predicted " + y_field)
    plt.title(f"Training set (y_train); R = {np.round(train_corr[0][1], 4)}")
    # Make axes of equal extents
    axes_lim = np.ceil(max(max(train_y), max(train_pred_y))) + 2
    plt.xlim((0, axes_lim))
    plt.ylim((0, axes_lim))
    plt.legend()

plotFeatureImportance()
plotTestY()
plotTrainY()
np.array(bgb_model.feature_names_in_)[sorted_idx]
np.array(bgb_model.feature_importances_)[sorted_idx]

# same procedure as above
data = pd.read_csv("csv/Aboveground Biomass_RS Model_5_year_Data.csv")  # for the 5-year averaged data
# data = pd.read_csv("csv/Aboveground Biomass_RS Model_Data.csv")
data.head()
# confirm column names first
cols = data.columns
# cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)
# data['ID'].value_counts()   # number of times same ID was sampled

# remove irrelevant columns for ML and determine X and Y variables
var_col =  [c for c in list(cols[18:-9]) + ['SRad'] if c not in ['dNDSI', 'AET', 'SWE', 'Wet_days', 'Cdef', 'Flow']]  # for the 5-year averaged data
# var_col =  list(cols[15:24]) + list(cols[-13:])
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

# bin the dataset based on BGB values
data[y_field].describe()
data['AGB_bin_class'] = 0
for value in range(50, 2450, 50):   # max rounded value is 2448
    mask = (data[y_field] > value) & (data[y_field] <= (value+50))
    data.loc[mask, 'AGB_bin_class'] = value//50
data['AGB_bin_class'].describe()

# split data into training (80%) and test data (20%) by IDs, random state ensures reproducibility
train_df = data.groupby('AGB_bin_class', group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=10))
test_data = data[~data.index.isin(train_df.index)]
# upsample the training dataset so that all bins have same amount of rows
max_size = train_df['AGB_bin_class'].value_counts().max()
train_data = (train_df.groupby('AGB_bin_class', group_keys=False)
    .apply(lambda x: resample(x, replace=True, n_samples=max_size, random_state=10)).reset_index(drop=True))
train_data['AGB_bin_class'].value_counts()

X_train, y_train = train_data.loc[:, var_col], train_data[y_field]
X_test, y_test = test_data.loc[:, var_col], test_data[y_field]

agb_model = GradientBoostingRegressor(learning_rate=0.13, max_depth=13, n_estimators=50, subsample=0.5, validation_fraction=0.2,
                                      n_iter_no_change=50, max_features='log2', verbose=1, random_state=10)
agb_84_model = GradientBoostingRegressor(loss="quantile", learning_rate=0.13, alpha=0.8413, max_depth=13, 
                                      n_estimators=50, subsample=0.5, validation_fraction=0.2, n_iter_no_change=50,  
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
feat_imp = agb_model.feature_importances_
sorted_idx = np.argsort(feat_imp)
pos = np.arange(sorted_idx.shape[0]) + 0.5

def plotFeatureImportance():
    plt.barh(pos, feat_imp[sorted_idx], align="center")
    plt.yticks(pos, np.array(agb_model.feature_names_in_)[sorted_idx])
    plt.title("Feature Importance")

plotFeatureImportance()
plotTestY()
plotTrainY()


with open('files/soil_models.pckl', 'wb') as f:   # there is also models.pckl
    pickle.dump([ghg_model, agb_model, bgb_model], f)
with open('files/sd_models.pckl', 'wb') as f:
    pickle.dump([ghg_84_model, agb_84_model, bgb_84_model], f)
