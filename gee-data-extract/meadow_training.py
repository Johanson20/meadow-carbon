# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""

import os, ee
import pandas as pd
import warnings
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

os.chdir("Code")
warnings.filterwarnings("ignore")

# Authenticate and initialize python access to Google Earth Engine
# ee.Authenticate()    # only use if you've never run this on your current computer before or loss GEE access
ee.Initialize()


def calculateIndices(image):
    # calculate and add indices from landsat band values
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndwi = image.normalizedDifference(['NIR', 'SWIR_1']).rename('NDWI')
    return image.addBands([ndvi, ndwi])


def maskCloudAndRename(image):
    # rename bands and mask out cloud based on bits in QA_pixel; then scale values
    image = image.rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'QA'])
    qa = image.select('QA')
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = dilated_cloud.And(cirrus).And(cloud).And(cloudShadow)
    image = image.updateMask(cloud_mask).select(['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2'])
    scaled_bands = image.multiply(2.75e-05).add(-0.2)
    return image.addBands(scaled_bands, overwrite=True)


def randomPolygonPoint(polygon):
    # extract coordinate of random point in a polygon (random seed ensures reproducibility) of result
    np.random.seed(10)
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(1000):  # Limit retries
        x, y = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            return [x, y]
    return None


# read in meadows shapefile and buffer by 1km; relevant datasets are landsat, elevation and flow accumulation data
epsg_crs = "EPSG:4326"
meadows = gpd.read_file("files/AllPossibleMeadows_2025-06-06.shp").to_crs(epsg_crs)
combined_meadows = meadows.union_all()
# Spatial resolutions: landsat is 30m, flow is 463.83m, elevation is 10.2m 
landsat9_collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])
landsat = landsat9_collection.filterDate('2022-07-01', '2022-07-31').map(maskCloudAndRename).map(calculateIndices)
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").resample('bilinear').reproject(crs="EPSG:32610", scale=30)
flow_acc_11 = ee.Image("WWF/HydroSHEDS/15ACC").resample('bilinear').reproject(crs="EPSG:32611", scale=30)
dem = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32610", scale=30)
dem_11 = ee.Image('USGS/3DEP/10m').select('elevation').reduceResolution(ee.Reducer.mean(), maxPixels=65536).reproject(crs="EPSG:32611", scale=30)
tpi = dem.subtract(dem.focalMean(5, 'square')).rename('TPI')
tpi_11 = dem_11.subtract(dem_11.focalMean(5, 'square')).rename('TPI')
slopeDem = ee.Terrain.slope(dem)
slopeDem_11 = ee.Terrain.slope(dem_11)

# initialize dataframe with relevant variables
meadow_data = pd.DataFrame(columns=['ID', 'Area_m2', 'Longitude', 'Latitude', 'BLue_mean', 'Blue_var', 'Green_mean', 'Green_var',
                           'NDVI_mean', 'NDVI_var', 'NDWI_mean', 'NDWI_var', 'NIR_mean', 'NIR_var', 'Red_mean', 'Red_var',
                           'SWIR_1_mean', 'SWIR_1_var', 'SWIR_2_mean', 'SWIR_2_var', 'TPI', 'Flow', 'Slope', 'IsMeadow'])

# iterate through each meadow: extract centroid's values and random non-meadow point
for meadowIdx in range(len(meadows)):
    feature = meadows.loc[meadowIdx, :]
    lon, lat = feature.geometry.centroid.coords[0]
    point = ee.Geometry.Point(lon, lat)
    
    # buffer meadow and extract random point in the buffer but outside all actual meadows
    buffer = feature.geometry.buffer(0.001)     # buffer units in latitude due to crs (~1km or ~1.11km)
    buffer_no_meadow = buffer.difference(combined_meadows)
    lon2, lat2 = randomPolygonPoint(buffer_no_meadow)
    non_meadow_point = ee.Geometry.Point(lon2, lat2)
    
    # read in values of relevant datasets for meadow's centroid and non-meadow random point
    landsat_images = landsat.filterBounds(point)
    np_landsat = landsat.filterBounds(non_meadow_point)
    if lon >= -120:   # corresponds to zone 32611
        flow = flow_acc_11.clip(point)
        slope_val = slopeDem_11.clip(point)
        tpi_val = tpi_11.clip(point)
        # same extraction for non-meadow point
        nflow = flow_acc_11.clip(non_meadow_point)
        nslope_val = slopeDem_11.clip(non_meadow_point)
        ntpi_val = tpi_11.clip(non_meadow_point)
    else:   # corresponds to zone 32610
        flow = flow_acc.clip(point)
        slope_val = slopeDem.clip(point)
        tpi_val = tpi.clip(point)
        # same extraction for non-meadow point
        nflow = flow_acc.clip(non_meadow_point)
        nslope_val = slopeDem.clip(non_meadow_point)
        ntpi_val = tpi.clip(non_meadow_point)
    
    # extract mean and variance of landsat and combine bands
    band_stats = landsat_images.reduce(ee.Reducer.mean().combine(ee.Reducer.variance(), sharedInputs=True))
    nband_stats = np_landsat.reduce(ee.Reducer.mean().combine(ee.Reducer.variance(), sharedInputs=True))
    combined_bands = band_stats.addBands([flow, slope_val, tpi_val])
    ncombined_bands = nband_stats.addBands([nflow, nslope_val, ntpi_val])
    # append band values to dataframe
    bandVal = combined_bands.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
    nbandVal = ncombined_bands.reduceRegion(ee.Reducer.mean(), non_meadow_point, 30).getInfo()
    meadow_data.loc[len(meadow_data)] = list(feature.values[:2]) + [lon, lat] + list(bandVal.values()) + ['Yes']
    meadow_data.loc[len(meadow_data)] = [feature.ID, 0, lon2, lat2] + list(nbandVal.values()) + ['No']
    
    if meadowIdx%20==0: print(meadowIdx, end=" ")

meadow_data.to_csv('csv/Real_and_false_meadows.csv', index=False)


import geemap
ee.Initialize()

# extract points of real and false meadows and convert to a feature collection for GEE classifier
meadow_points = gpd.GeoDataFrame({'MeadowId': meadow_data['ID'], 'Longitude': meadow_data['Longitude'], 'Latitude': meadow_data['Latitude'], 'IsMeadow': [1 if c == 'Yes' else 0 for c in meadow_data['IsMeadow']]}, geometry = [Point(x,y) for x,y in zip(meadow_data['Longitude'], meadow_data['Latitude'])], crs=epsg_crs)
featureColl = geemap.geopandas_to_ee(meadow_points, geodesic=False)
# combine relevant imagery to be used as predictors (independent variables)
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC")
dem = ee.Image('USGS/3DEP/10m').select('elevation')
tpi = dem.subtract(dem.focalMean(5, 'square')).rename('TPI')
slopeDem = ee.Terrain.slope(dem)
predictors = landsat.median().addBands([flow_acc, slopeDem, tpi])
# ensure feature collection has a label for classification ('IsMeadow') of type Float
training = predictors.sampleRegions(collection=featureColl, properties=['IsMeadow'], scale=30, geometries=False)

# tune hyperparameter for number of trees
best_tree = {}
for ntree in range(10, 300, 5):
    classifier = ee.Classifier.smileRandomForest(numberOfTrees=ntree, seed=10).train(features=training, classProperty='IsMeadow', inputProperties=predictors.bandNames())
    classified = predictors.classify(classifier)
    validation = classified.sampleRegions(collection=featureColl, properties=['IsMeadow'], scale=30, geometries=False)
    acc = validation.errorMatrix('IsMeadow', 'classification').accuracy().getInfo()
    best_tree[ntree] = acc
    print(ntree, "=", acc,  end=' ')

# use best result
ntrees = 50     # max(best_tree, key=best_tree.get)
classifier = ee.Classifier.smileRandomForest(numberOfTrees=ntrees, seed=10).train(features=training, classProperty='IsMeadow', inputProperties=predictors.bandNames())
classified = predictors.classify(classifier)

# extract predictions of classes and append predictions to dataset for IDs available
classes = classified.sampleRegions(collection=featureColl, properties=['IsMeadow'], scale=30, geometries=False)
classes = ee.data.computeFeatures({'expression': classes, 'fileFormat': 'PANDAS_DATAFRAME'})
meadow_points['RFPredictedClass'] = classes['classification']
realMeadows = meadows[meadows['ID'].isin(set(meadow_data['ID']))]
realMeadows['RFPredictedClass'] = meadow_points.groupby('MeadowId')['RFPredictedClass'].first()

'''meadow_data.dropna(inplace=True)
meadow_data.drop_duplicates(inplace=True)
meadow_data.reset_index(drop=True, inplace=True)'''
meadow_points.to_file("files/Meadow_Points.shp", driver="ESRI Shapefile")
realMeadows.to_file("files/PredictedMeadows.shp", driver="ESRI Shapefile")
meadows = None