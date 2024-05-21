# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""


import os, ee
import pandas as pd
import geopandas as gpd
os.chdir("Code")

# ee.Authenticate()
ee.Initialize()

shapefile = gpd.read_file("files/AllPossibleMeadows_2024-02-12.shp")

feature = shapefile.iloc[4].geometry
list(shapefile.columns)
minx, miny, maxx, maxy = feature.bounds
nlayers = len(shapefile)

landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate('2022-07-01', '2022-07-31')
# landsat9_collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate('2022-07-01', '2022-07-31')

def calculate_indices(image):
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    return image.addBands([ndvi, ndwi])

out_df = pd.DataFrame(columns=['ID', 'Area_m2', 'Area_km2', 'NDVI', 'NDWI'])
rowId = 0

for index, row in shapefile.iterrows():
    feature = row['geometry']

    if feature.geom_type == 'Polygon':
        shapefile_bbox = ee.Geometry.Polygon(list(feature.exterior.coords))
    elif feature.geom_type == 'MultiPolygon':
        shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geoms))
    
    landsat_image = landsat8_collection.filterBounds(shapefile_bbox).sort("CLOUD_COVER").first()
    meadow_image = calculate_indices(landsat_image)
    
    indices = meadow_image.select(['NDVI', 'NDWI']).reduceRegion(ee.Reducer.mean(), shapefile_bbox, 30).getInfo()
    ndvi_number, ndwi = list(indices.values())
    
    if not ndvi_number or not ndwi or (ndvi_number < 0.2 and ndwi < 0.5):
        val = list(row.values)[:3]
        val += [ndvi_number, ndwi]
        out_df.loc[rowId] = val
        rowId += 1
    
    if index%100==0: print(index, end=" ")
    
shapefile = None
out_df.to_csv('csv/False_meadows_2022.csv', index=False)
