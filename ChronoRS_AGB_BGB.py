import os
import ee
import pandas as pd
os.chdir("Code")    # adjust directory

# read csv file and convert dates from strings to datetime
data = pd.read_csv("ChronoRS_AGB.csv")
data.head()

# Authenticate and Initialize the Earth Engine API
#ee.Authenticate()
ee.Initialize()

landsat8_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

def addB5(image):
    return image.set('B5_value', image.select('B5').reduceRegion(ee.Reducer.mean(), point, 30).get('B5'))


NIR = []

# populate bands by applying above functions for each pixel in dataframe
for i in range(data.shape[0]):
    x, y = data.loc[i, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    year = data.loc[i, 'SampleDate'].split("/")[2]

    temporal_filtered = landsat8_collection.filterDate(year+"-01-01", year+"-12-31")
    spatial_filtered = temporal_filtered.filterBounds(point)
    landsat8_with_b5 = spatial_filtered.map(addB5)
    sorted_collection = landsat8_with_b5.sort('B5_value', False)
    NIR_value = sorted_collection.first().getInfo()['properties']['B5_value']
    
    NIR.append(NIR_value)
    
    if i%20 == 0: print(i, end=' ')
# 20.342065 seconds for AGB

# checks if they are all cloud free (should equal data.shape[0])
ids = [x for x in NIR if x]
len(ids)

# update and display first 10 rows of dataframe
data['NIR'] = NIR
data.head(10)

# write updated dataframe to new csv file
data.to_csv('ChronoRS_AGB_NIR.csv', index=False)


# REPEAT same for BGB
data = pd.read_csv("ChronoRS_BGB.csv")
data['SampleDate'] = pd.to_datetime(data['SampleDate'], format = '%m/%d/%y')
data.head()

NIR = []

for i in range(data.shape[0]):
    x, y = data.loc[i, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    year = data.loc[i, 'SampleDate'].strftime('%Y')

    temporal_filtered = landsat8_collection.filterDate(year+"-01-01", year+"-12-31")
    spatial_filtered = temporal_filtered.filterBounds(point)
    landsat8_with_b5 = spatial_filtered.map(addB5)
    sorted_collection = landsat8_with_b5.sort('B5_value', False)
    NIR_value = sorted_collection.first().getInfo()['properties']['B5_value']
    
    NIR.append(NIR_value)
    
    if i%20 == 0: print(i, end=' ')
# 45.063906 seconds for BGB

# checks if they are all cloud free (should equal data.shape[0])
ids = [x for x in NIR if x]
len(ids)

# update and display first 10 rows of dataframe
data['NIR'] = NIR
data.head(10)

# write updated dataframe to new csv file
data.to_csv('ChronoRS_BGB_NIR.csv', index=False)
