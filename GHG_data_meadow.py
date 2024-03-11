import os
import ee
import pandas as pd
os.chdir("Code")    # adjust directory

# read csv file and convert dates from strings to datetime
data = pd.read_csv("GHG_Data_Sample.csv")
data['Date'] = pd.to_datetime(data['Date'], format = '%m/%d/%y')
data.head()

# Authenticate and Initialize the Earth Engine API
#ee.Authenticate()
ee.Initialize()


# Function to mask clouds
def maskClouds(image):
    quality = image.select('QA_PIXEL')
    cloud = quality.bitwiseAnd(1 << 5).eq(0)    # mask out cloudy pixels
    clear = quality.bitwiseAnd(1 << 4).eq(0)     # mask out cloud shadow
    return image.updateMask(cloud).updateMask(clear)

# Calculates absolute time difference (in days) from a target date, in which the images are acquired
def calculate_time_difference(image):
    time_difference = ee.Number(image.date().difference(target_date, 'day')).abs()
    return image.set('time_difference', time_difference)


# Function to extract cloud free band values per pixel from landsat 8 or landsat 7
def getBandValues(landsat_collection, point, target_date, bufferDays = 30, landsatNo = 8):
    # filter landsat images by location
    spatial_filtered = landsat_collection.filterBounds(point)
    # filter the streamlined images by dates +/- a certain number of days
    temporal_filtered = spatial_filtered.filterDate(ee.Date(target_date).advance(-bufferDays, 'day'), ee.Date(target_date).advance(bufferDays, 'day'))
    # apply cloud mask and sort images in the collection
    cloud_free_images = temporal_filtered.map(maskClouds)
    # Map the ImageCollection over time difference and sort by that property
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
        properties = nearest_image.getInfo()['properties']
        band_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    
    return [list(band_values.values()), properties['time_difference'], properties['DATE_ACQUIRED'], properties['SCENE_CENTER_TIME']]


# reads surface reflectance values of Tier 1 collections of landsat 8 and 7
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
landsat7_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")

# define arrays to store band values and landsat information
Blue, Green, Red, NIR, SWIR_1, SWIR_2 = [], [], [], [], [], []
acq_date, acq_time, driver, time_diff = [], [], [], []

# populate bands by applying above functions for each pixel in dataframe
for id in range(data.shape[0]):
    x, y = data.loc[id, ['Long', 'Lat']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[id, 'Date'].strftime('%Y-%m-%d')
    # 30 day radius used to search for cloud-free images
    vxn = 8
    band_values, t_diff, date_epoch, time_epoch = getBandValues(landsat8_collection, point, target_date, 30)
    if not band_values[0]:
        vxn = 7
        band_values, t_diff, date_epoch, time_epoch = getBandValues(landsat7_collection, point, target_date, 30, 7)
        # 60 day radius used to find more cloud-free images
        if not band_values[0]:
            vxn = 8
            print(id, "Searching Landsat 8 collection with 60-day search radius")
            band_values, t_diff, date_epoch, time_epoch = getBandValues(landsat8_collection, point, target_date, 60)
            if not band_values[0]:
                vxn = 7
                print(id, "Searching Landsat 7 collection with 60-day search radius")
                band_values, t_diff, date_epoch, time_epoch = getBandValues(landsat7_collection, point,
                                                                 target_date, 60, 7)
                
    Blue.append(band_values[0])
    Green.append(band_values[1])
    Red.append(band_values[2])
    NIR.append(band_values[3])
    SWIR_1.append(band_values[4])
    SWIR_2.append(band_values[5])
    driver.append(vxn)
    time_diff.append(t_diff)
    acq_date.append(date_epoch)
    acq_time.append(time_epoch.split('.')[0])
    
    if id%100 == 0: print(id, end=' ')


data['Blue'] = Blue
data['Green'] = Green
data['Red'] = Red
data['NIR'] = NIR
data['SWIR_1'] = SWIR_1
data['SWIR_2'] = SWIR_2
data['Acquisition_Date'] = acq_date
data['Acquisition_Time'] = acq_time
data['Driver'] = driver
data['Days_of_data_acquisition_offset'] = time_diff

# display first 10 rows of updated dataframe
data.head(10)

# checks how many pixels are cloud free (non-null value);
# all bands would be simultaneously cloud-free or not
ids = [x for x in Blue if x]

# write updated dataframe to new csv file
data.to_csv('GHG_Data_Sample_Bands.csv', index=False)


data = pd.read_csv("GHG_Data_Sample_Bands.csv")
data.head()

min_temp, max_temp = [], []
# loop through and extract min and max temperatures from gridmet
for id in range(data.shape[0]):
    x, y = data.loc[id, ['Long', 'Lat']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[id, 'Date']
    gridmet_filtered = gridmet.filterDate(ee.Date(target_date).advance(-16, 'day'), ee.Date(target_date).advance(16, 'day'))
    bands = ee.Image(gridmet_filtered.first()).select(['tmmn', 'tmmx'])
    temperature_values = bands.reduceRegion(ee.Reducer.mean(), point, 4000).getInfo()
    tmin = temperature_values['tmmn']
    tmax = temperature_values['tmmx']
    
    min_temp.append(tmin)
    max_temp.append(tmax)
    
    if id%100 == 0: print(id, end=' ')

data['Minimum_temperature'] = min_temp
data['Maximum_temperature'] = max_temp

data.to_csv('GHG_Data_Sample_Bands_Temp.csv', index=False)

'''
# TESTING Landsat usage from google earth engine
# Define the point coordinates where you want to find Landsat images
x, y = data.loc[1, ['Long', 'Lat']]
point = ee.Geometry.Point(x, y)
# read the date of the second row of dataframe, and format as string
target_date = data.loc[1, 'Date'].strftime('%Y-%m-%d')
# apply cloud masking to landsat 8 collection
cloud_free_images = landsat8_collection.map(maskClouds)
# sort cloud free images and take least cloudy one
nearest_image = ee.Image(cloud_free_images.sort('cloud').first())


import geemap

# display GEE map as standard false colour composite
map_l8 = geemap.Map(center=[y,x], zoom=10)  # note that latitude comes first here
image_viz_params = {'bands': ['B5', 'B4', 'B3'], 'min': 0, 'max': 0.5, 'gamma': [0.95, 1.1, 1]}
# Add the image layer to the map and display it.
map_l8.add_layer(nearest_image, image_viz_params, 'false color composite')
display(map_l8)'''