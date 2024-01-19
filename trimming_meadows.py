import os, ee
from osgeo import ogr, osr
import pandas as pd
from shapely.geometry import mapping, box
os.chdir("Code")

ee.Initialize()

shapefile = ogr.Open("AllPossibleMeadows_2024-01-10.shp")
layer = shapefile.GetLayer()
spatref = layer.GetSpatialRef()

wkt = spatref.ExportToWkt()
xycrs = osr.SpatialReference()
xycrs.ImportFromWkt(wkt)
latlon_crs = osr.SpatialReference()
latlon_crs.ImportFromEPSG(32611)  # WGS84
transform = osr.CoordinateTransformation(xycrs, latlon_crs)

'''landsat_image = ee.Image('LANDSAT/LC08/C01/T1/LC08_044034_20140318')
im1 = landsat_image.select('B3')
latlon_crs = im1.projection().getInfo()'''

feature = layer.GetFeature(0)
[field.GetName() for field in layer.schema]
feature.GetField('ID')
minx, maxx, miny, maxy = layer.GetExtent()
nlayers = layer.GetFeatureCount()

landsat8_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

def calculate_ndvi(image):
    ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

out_df = pd.DataFrame(columns=['ID', 'Area_m2'])
rowId = 0

for feature in layer:
    geometry = feature.GetGeometryRef()
   
    # Convert the polygon to a bounding box for GEE Landsat image filtering
    bbox = box(*geometry.GetEnvelope())
    bbox_coords = list(mapping(bbox)['coordinates'][0])
    
    points = [(minx, miny), (maxx, maxy)]
    coords = [transform.TransformPoint(x, y) for x, y in points]
    min_lat, min_lon = coords[0][:2]
    max_lat, max_lon = coords[1][:2]

    bbox_ee = ee.Geometry.Polygon(bbox_coords)
    landsat_images = landsat8_collection.filterBounds(bbox_ee) \
                                       .filterDate('2021-07-01', '2021-07-31')
    meadow_image = landsat_images.sort("CLOUD_COVER").first()        

    # Sample NDVI at the point
    ndvi_value = meadow_image.select('NDVI') \
        .reduceRegion(reducer=ee.Reducer.first(), geometry=bbox_ee, scale=30)
    
    # Get the NDVI value as a number
    ndvi_number = ndvi_value.getNumber('NDVI')
    
    if ndvi_number < 0.2:
        out_df.loc[rowId] = [feature.GetField('ID'), feature.GetField('Area_m2')]
        rowId += 1
    
layer = None
out_df.to_csv('False_meadows.csv', index=False)