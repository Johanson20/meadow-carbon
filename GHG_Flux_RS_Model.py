import os
import ee
import pandas as pd
os.chdir("Code")    # adjust directory

# read csv file and convert dates from strings to datetime
data = pd.read_csv("GHG Flux_RS Model.csv")
sum(data['Longitude'].isna())
data.head()

# Authenticate and Initialize the Earth Engine API
#ee.Authenticate()
ee.Initialize()


# reads Landsat data, flow accumulation, gridmet temperature and DEM data (for slope and elevation)
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
landsat7_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
dem = ee.Image('USGS/SRTMGL1_003').select('elevation')


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
        time_diff = nearest_image.getInfo()['properties']['time_difference']
        band_values = bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    
    return [list(band_values.values()), time_diff]


# define arrays to store band values and landsat information
Blue, Green, Red, NIR, SWIR_1, SWIR_2 = [], [], [], [], [], []
flow, slope, elevation, driver, time_diff = [], [], [], [], []
min_temp, max_temp = [], []

# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
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
    
    # extract Landsat band values
    vxn = 8
    band_values, t_diff = getBandValues(landsat8_collection, point, target_date, 30)
    if not band_values[0]:
        vxn = 7
        band_values, t_diff = getBandValues(landsat7_collection, point, target_date, 30, 7)
        # 60 day radius used to find more cloud-free images
        if not band_values[0]:
            vxn = 8
            print(idx, "Searching Landsat 8 collection with 60-day search radius")
            band_values, t_diff = getBandValues(landsat8_collection, point, target_date, 60)
            if not band_values[0]:
                vxn = 7
                print(idx, "Searching Landsat 7 collection with 60-day search radius")
                band_values, t_diff = getBandValues(landsat7_collection, point, target_date, 60, 7)
    
    # append vaues to different lists
    Blue.append(band_values[0])
    Green.append(band_values[1])
    Red.append(band_values[2])
    NIR.append(band_values[3])
    SWIR_1.append(band_values[4])
    SWIR_2.append(band_values[5])
    flow.append(flow_value)
    elevation.append(elev)
    slope.append(slope_value)
    driver.append(vxn)
    time_diff.append(t_diff)
    min_temp.append(tmin)
    max_temp.append(tmax)

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
data['Driver'] = driver
data['Days_of_data_acquisition_offset'] = time_diff
data['Minimum_temperature'] = min_temp
data['Maximum_temperature'] = max_temp

# display first 10 rows of updated dataframe
data.head(10)

# checks how many pixels are cloud free (non-null value);
# all bands would be simultaneously cloud-free or not
ids = [x for x in Blue if x]
len(ids)

# write updated dataframe to new csv file
data.to_csv('GHG_Flux_RS_Model_Data.csv', index=False)
