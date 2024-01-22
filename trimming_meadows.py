import os, ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, box
os.chdir("Code")

ee.Initialize()

shapefile = gpd.read_file("AllPossibleMeadows_2024-01-19.shp")

feature = shapefile.geometry.iloc[0]
list(shapefile.columns)
minx, miny, maxx, maxy = feature.bounds
nlayers = len(shapefile)

landsat8_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

def calculate_ndvi(image):
    ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

out_df = pd.DataFrame(columns=['ID', 'Area_m2', 'NDVI'])
rowId = 0

for index, row in shapefile.iterrows():
    geometry = row['geometry']
   
    # Convert the polygon to a bounding box for GEE Landsat image filtering
    bbox = box(*geometry.bounds)
    bbox_coords = list(mapping(bbox)['coordinates'][0])
    
    bbox_ee = ee.Geometry.Polygon(bbox_coords)
    landsat_images = landsat8_collection.filterBounds(bbox_ee).filterDate('2021-07-01', '2021-07-31')
    meadow_image = landsat_images.map(calculate_ndvi).sort("CLOUD_COVER").first()        

    # Sample NDVI at the region and extract value
    ndvi_value = meadow_image.select('NDVI').reduceRegion(ee.Reducer.mean(), bbox_ee, scale=30)
    ndvi_number = ndvi_value.getNumber('NDVI').getInfo()
    
    if not ndvi_number or ndvi_number < 0.2:
        val = list(row.values)[:2]
        val.append(ndvi_number)
        out_df.loc[rowId] = val
        rowId += 1
    
    if index%100==0: print(index, end=" ")
    
shapefile = None
out_df.to_csv('False_meadows.csv', index=False)