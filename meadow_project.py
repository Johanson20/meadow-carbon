import os
import ee
import geemap
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
    quality = image.select('pixel_qa')
    cloud = quality.bitwiseAnd(1 << 5).eq(0)    # mask out cloud shadow
    clear = quality.bitwiseAnd(1 << 4).eq(0)     # mask out cloud
    return image.updateMask(cloud).updateMask(clear)


# Function to extract cloud free band values per pixel from landsat 8 or landsat 7
def getBandValues(landsat_collection, point, target_date, bufferDays = 30, landsatNo = 8):
    # filter landsat images by location
    spatial_filtered = landsat_collection.filterBounds(point)
    # filter the streamlined images by dates +/- a certain number of days
    temporal_filtered = spatial_filtered.filterDate(ee.Date(target_date).advance(-bufferDays, 'day'), ee.Date(target_date).advance(bufferDays, 'day'))
    # apply cloud mask and sort images in the collection
    cloud_free_images = temporal_filtered.map(maskClouds)
    sorted_collection = cloud_free_images.sort('cloud')
    image_list = sorted_collection.toList(sorted_collection.size())
    noImages = image_list.size().getInfo()
    nImage, band_values = 0, {'B2': None}
    
    # repeatedly check for cloud free pixels (non-null value) in landsat 8, or checks in landsat 7
    while band_values['B2'] == None and nImage < noImages:
        nearest_image = ee.Image(image_list.get(nImage))
        nImage += 1
        if landsatNo == 7:
            bands = nearest_image.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7'])
        else:
            bands = nearest_image.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
        band_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    
    return band_values


# reads surface reflectance values of Tier 1 collections of landsat 8 and 7
landsat8_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
landsat7_collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')


# define arrays to store band values
Blue, Green, Red, NIR, SWIR_1, SWIR_2 = [], [], [], [], [], []

# populate bands by applying above functions for each pixel in dataframe
for id in range(data.shape[0]):
    x, y = data.loc[id, ['Long', 'Lat']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[id, 'Date'].strftime('%Y-%m-%d')
    # 60 day radius used to find more cloud-free images
    band_values = getBandValues(landsat8_collection, point, target_date, 60)
    if band_values['B2']:
        Blue.append(band_values['B2'])
        Green.append(band_values['B3'])
        Red.append(band_values['B4'])
        NIR.append(band_values['B5'])
        SWIR_1.append(band_values['B6'])
        SWIR_2.append(band_values['B7'])
    else:
        band_values = getBandValues(landsat7_collection, point, target_date, 60, 7)
        Blue.append(band_values['B1'])
        Green.append(band_values['B2'])
        Red.append(band_values['B3'])
        NIR.append(band_values['B4'])
        SWIR_1.append(band_values['B5'])
        SWIR_2.append(band_values['B7'])
    
    if id % 100 == 0:
        print(id, end = ' ')

data['Blue'] = Blue
data['Green'] = Green
data['Red'] = Red
data['NIR'] = NIR
data['SWIR_1'] = SWIR_1
data['SWIR_2'] = SWIR_2

# display first 10 rows of updated dataframe
data.head(10)

# checks how many pixels are cloud free (non-null value);
# all bands would be simultaneously cloud-free or not
len([x for x in Blue if x])

# write updated dataframe to new csv file
data.to_csv('GHG_Data_Sample_Bands.csv', index=False)


# TESTING Landsat usage from google earth engine
# Define the point coordinates where you want to find Landsat images
x, y = data.loc[1, ['Lat', 'Long']]
point = ee.Geometry.Point(x, y)

# read the date of the second row of dataframe, and format as string
target_date = data.loc[1, 'Date'].strftime('%Y-%m-%d')

# apply cloud masking to landsat 8 collection
cloud_free_images = landsat8_collection.map(maskClouds)

# sort cloud free images and take least cloudy one
nearest_image = ee.Image(cloud_free_images.sort('cloud').first())

# read specific bands from above image
bands = nearest_image.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
# extract values of those bands for a single pixel in the selected landsat image
band_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()

print('Band Values:', band_values)

print('Nearest image:', nearest_image)

# Retrieve Landsat image metadata
image_info = nearest_image.getInfo()
print('Image metadata:', image_info)

# display GEE map as standard false colour composite
map_l8 = geemap.Map(center=[y,x], zoom=10)  # note that latitude comes first here
image_viz_params = {'bands': ['B5', 'B4', 'B3'], 'min': 0, 'max': 0.5, 'gamma': [0.95, 1.1, 1]}

# Add the image layer to the map and display it.
map_l8.add_layer(nearest_image, image_viz_params, 'false color composite')
display(map_l8)