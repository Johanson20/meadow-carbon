import os, ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, box
os.chdir("Code")

# ee.Authenticate()
ee.Initialize()

shapefile = gpd.read_file("AllPossibleMeadows_2024-01-23.2.shp")

feature = shapefile.iloc[4].geometry
list(shapefile.columns)
minx, miny, maxx, maxy = feature.bounds
nlayers = len(shapefile)

landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate('2022-07-01', '2022-07-31')
# landsat9_collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate('2022-07-01', '2022-07-31')

def calculate_ndvi(image):
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    return image.addBands(ndvi)

out_df = pd.DataFrame(columns=['ID', 'Area_m2', 'Area_km2', 'NDVI'])
rowId = 0

for index, row in shapefile.iterrows():
    feature = row['geometry']
   
    # Convert the polygon to a bounding box for GEE Landsat image filtering
    bbox = box(*feature.bounds)
    bbox_coords = list(mapping(bbox)['coordinates'][0])
    
    bbox_ee = ee.Geometry.Polygon(bbox_coords)
    
    landsat_images = landsat8_collection.filterBounds(bbox_ee)
    meadow_image = calculate_ndvi(landsat_images.sort("CLOUD_COVER").first())
    
    if feature.geom_type == 'Polygon':
        shapefile_bbox = ee.Geometry.Polygon(list(feature.exterior.coords))
    elif feature.geom_type == 'MultiPolygon':
        shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geoms))
    
    landsat_masked = meadow_image.updateMask(meadow_image.clip(shapefile_bbox))
    # Sample NDVI at the region and extract value
    ndvi_value = landsat_masked.select('NDVI').reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox_ee, scale=30)
    ndvi_number = ndvi_value.getNumber('NDVI').getInfo()
    
    if not ndvi_number or ndvi_number < 0.2:
        val = list(row.values)[:3]
        val.append(ndvi_number)
        out_df.loc[rowId] = val
        rowId += 1
    
    if index%100==0: print(index, end=" ")
    
shapefile = None
out_df.to_csv('False_meadows.csv', index=False)