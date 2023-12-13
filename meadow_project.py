import os
import ee
import pandas as pd
os.chdir("C:/Users/Johanson Onyegbula/OneDrive - University of Nevada, Reno/Documents/Point Blue/Code")

data = pd.read_csv("GHG_Data_Sample.csv")
data['Date'] = pd.to_datetime(data['Date'], format = '%m/%d/%y')
data.head()

# Authenticate and Initialize the Earth Engine API
#ee.Authenticate()
ee.Initialize()

# Define the point coordinates where you want to find Landsat images
x, y = data.loc[1, ['Lat', 'Long']]
point = ee.Geometry.Point(x, y)

target_date = data.loc[1, 'Date'].strftime('%Y-%m-%d')

# Function to mask clouds
def maskClouds(image):
    quality = image.select('pixel_qa')
    cloud = quality.bitwiseAnd(1 << 5).eq(0)    # mask out cloud shadow
    clear = quality.bitwiseAnd(1 << 4).eq(0)     # mask out cloud
    return image.updateMask(cloud).updateMask(clear)

def calculateCloudScore(image):
    cloud = ee.Algorithms.Landsat.simpleCloudScore(image).select('cloud')
    return image.addBands(cloud)

# Filter Landsat collection based on location and date
landsat_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
    .filterBounds(point) \
    .filterDate(ee.Date(target_date).advance(-7, 'day'), ee.Date(target_date).advance(7, 'day'))

cloud_free_images = landsat_collection.map(maskClouds).map(calculateCloudScore)

nearest_image = ee.Image(landsat_collection.sort('cloud').first())  # 'system:time_start'

bands = nearest_image.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
band_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()

print('Band Values:', band_values)

# Print the nearest image
print('Nearest image:', nearest_image)

# Retrieve Landsat image metadata
image_info = nearest_image.getInfo()
print('Image metadata:', image_info)
