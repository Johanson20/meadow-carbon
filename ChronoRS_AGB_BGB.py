# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""


import os
import ee
import pandas as pd
os.chdir("Code")    # adjust directory

# read csv file and convert dates from strings to datetime
data = pd.read_csv("csv/ChronoRS_AGB.csv")
data.head()

# Authenticate and Initialize the Earth Engine API
#ee.Authenticate()
ee.Initialize()
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")


# add B5 band value explicitly to properties of landsat
def addB5(image):
    return image.set('B5_value', image.select('B5').reduceRegion(ee.Reducer.mean(), point, 30).get('B5'))


# extract unique years and create a dictionary of landsat data for each year
years = set(x.split("/")[2] for x in data.loc[:, 'SampleDate'])
landsat = {}
for year in years:
    landsat[year] = landsat8_collection.filterDate(year+"-01-01", year+"-12-31")


NIR, peak_dates = [], []
# populate bands by applying above functions for each pixel in dataframe
for i in range(data.shape[0]):
    x, y = data.loc[i, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    year = data.loc[i, 'SampleDate'].split("/")[2]

    # filter landsat by area and year, and sort by value of added B5 band
    spatial_filtered = landsat[year].filterBounds(point)
    landsat8_with_b5 = spatial_filtered.map(addB5)
    sorted_collection = landsat8_with_b5.sort('B5_value', False)
    properties = sorted_collection.first().getInfo()['properties']
    
    peak_dates.append(properties['SENSING_TIME'][:10])
    NIR.append(properties['B5_value'])
    
    if i%20 == 0: print(i, end=' ')

# checks if they are all cloud free (should equal data.shape[0])
ids = [x for x in NIR if x]
len(ids)

# update and display first 10 rows of dataframe
data['NIR'] = NIR
data['peak_date'] = peak_dates
data.head(10)

# write updated dataframe to new csv file
data.to_csv('csv/ChronoRS_AGB_NIR.csv', index=False)


# REPEAT same for BGB
data = pd.read_csv("csv/ChronoRS_BGB.csv")
data['SampleDate'] = pd.to_datetime(data['SampleDate'], format = '%m/%d/%y')
data.head()

years = set(x.split("/")[2] for x in data.loc[:, 'SampleDate'].strftime('%Y'))
landsat = {}
for year in years:
    landsat[year] = landsat8_collection.filterDate(year+"-01-01", year+"-12-31")

NIR, peak_dates = [], []

for i in range(data.shape[0]):
    x, y = data.loc[i, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    year = data.loc[i, 'SampleDate'].strftime('%Y')

    spatial_filtered = landsat[year].filterBounds(point)
    landsat8_with_b5 = spatial_filtered.map(addB5)
    sorted_collection = landsat8_with_b5.sort('B5_value', False)
    properties = sorted_collection.first().getInfo()['properties']
    
    peak_dates.append(properties['SENSING_TIME'][:10])
    NIR.append(properties['B5_value'])
    
    if i%20 == 0: print(i, end=' ')

# checks if they are all cloud free (should equal data.shape[0])
ids = [x for x in NIR if x]
len(ids)

# update and display first 10 rows of dataframe
data['NIR'] = NIR
data['peak_date'] = peak_dates
data.head(10)

# write updated dataframe to new csv file
data.to_csv('csv/ChronoRS_BGB_NIR.csv', index=False)
