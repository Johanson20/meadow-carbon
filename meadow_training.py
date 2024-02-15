import os, ee
import pandas as pd
import geopandas as gpd
os.chdir("Code")

# Authenticate and initialize python access to Google Earth Engine
# ee.Authenticate    # only use if you've never run this on your current computer before or loss GEE access
ee.Initialize()

# read in shapefile, landsat and flow accumulation data
shapefile = gpd.read_file("AllPossibleMeadows_2024-02-12.shp")
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate('2022-07-01', '2022-07-31')
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')

# Function to mask clouds
def maskClouds(image):
    quality = image.select('QA_PIXEL')
    cloud = quality.bitwiseAnd(1 << 5).eq(0)    # mask out cloudy pixels
    clear = quality.bitwiseAnd(1 << 4).eq(0)     # mask out cloud shadow
    return image.updateMask(cloud).updateMask(clear)

# initialize dataframe with relevant variables
df = pd.DataFrame(columns=['ID', 'longitude', 'latitude', 'flow_accumulation', 'QA_PIXEL_mean', 'QA_PIXEL_variance',
                           'B1_mean', 'B1_variance', 'B2_mean', 'B2_variance', 'B3_mean', 'B3_variance', 'B4_mean', 
                           'B4_variance', 'B5_mean', 'B5_variance', 'B6_mean', 'B6_variance', 'B7_mean', 'B7_variance'])

# iterate through each polygon of shapefile
for index, row in shapefile.iterrows():
    feature = row.geometry
    lon, lat = feature.centroid.coords[0]
    
    # extract polygon bounding coordinates (they are either single and whole or multiple and disjoint)
    if feature.geom_type == 'Polygon':
        shapefile_bbox = ee.Geometry.Polygon(list(feature.exterior.coords))
    elif feature.geom_type == 'MultiPolygon':
        shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geoms))
    
    # read in flow and landsat data for every 30m square (note that flow acc actually has 463.83m resolution)
    flow = flow_acc.reduceRegion(ee.Reducer.mean(), shapefile_bbox, 30).getInfo()
    landsat_images = landsat8_collection.filterBounds(shapefile_bbox).map(maskClouds)
    
    # extract mean and variance of landsat and read relevant band names
    band_stats = landsat_images.reduce(ee.Reducer.mean().combine(ee.Reducer.variance(), sharedInputs=True))
    relevant_bands = band_stats.bandNames().getInfo()[:14] + band_stats.bandNames().getInfo()[34:36]
    
    # compute band values of landsat (try block handles failure in cases of data overload for huge meadows)
    try:
        bandVal = band_stats.select(relevant_bands).reduceRegion(ee.Reducer.mean(), shapefile_bbox, 30).getInfo()
        df.loc[index] = [row.values[0], lon, lat, flow['b1']] + [s for s in bandVal.values()]
    except:
        bandVal = band_stats.select(relevant_bands[:8]).reduceRegion(ee.Reducer.mean(), shapefile_bbox, 30).getInfo()
        valu = band_stats.select(relevant_bands[8:]).reduceRegion(ee.Reducer.mean(), shapefile_bbox, 30).getInfo()
        df.loc[index] = [row.values[0], lon, lat, flow['b1']] + [s for s in list(valu.values())[:2]] + [s for s in bandVal.values()] + [s for s in list(valu.values())[2:]]
    
    if index%100==0: print(index, end=" ")

shapefile = None
df.to_csv('All_meadows_2022.csv', index=False)
