# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:03:25 2024

@author: jonyegbula
"""

import os
import ee
import numpy as np
import pickle
import warnings
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from geocube.api.core import make_geocube
import pyproj
import geemap
import contextlib
from osgeo import gdal

mydir = "Code"
os.chdir(mydir)

# Suppress warnings but show errors
warnings.filterwarnings("ignore")
def gdal_warning_handler(err_class, err_num, err_msg):
    if err_class == gdal.CE_Warning:
        pass
    else:
        gdal.ErrorHandler(err_class, err_num, err_msg)
gdal.PushErrorHandler(gdal_warning_handler)

# Authenticate and initialize python access to Google Earth Engine
# ee.Authenticate()    # only use if you've never run this on your current computer before or loss GEE access
ee.Initialize()

# read in shapefile, landsat and flow accumulation data
shapefile = gpd.read_file("files/AllPossibleMeadows_2024-02-12.shp")
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate('2018-10-01', '2019-10-01')
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select('b1')
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate('2018-10-01', '2019-10-01').select(['pr', 'tmmn', 'tmmx'])
dem = ee.Image('USGS/3DEP/10m').select('elevation')
slope = ee.Terrain.slope(dem)
daymet = ee.ImageCollection("NASA/ORNL/DAYMET_V4").select('swe')

def maskImage(image):
    qa = image.select('QA_PIXEL')
    # mask out cloud based on bits in QA_pixel
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloudShadow = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = dilated_cloud.And(cirrus).And(cloud).And(cloudShadow)
    return image.updateMask(cloud_mask)

def geotiffToCsv(input_raster, bandnames, crs):
    # creates a dataframe of unique columns (hence combines repeating band names)
    pd.set_option("display.precision", 20)
    geotiff = gdal.Open(input_raster)
    nBands = geotiff.RasterCount
    ncol = len(set(bandnames))
    out_csv = pd.DataFrame()
    n = ((nBands - ncol)//9 + 1)
    
    # There are 9 (repeating) landsat and gridmet bands extracted in total
    for col in range(1, 10):
        values = []
        for band in [col] + list(range(ncol+col, nBands+1, 9)):
            bandValues = geotiff.GetRasterBand(band).ReadAsArray()
            for value in bandValues:
                values.extend(value)
        out_csv[bandnames[col-1]] = values
    
    # repeat the unique columns throughout length of dataframe
    for col in range(10, 14):
        bandValues = geotiff.GetRasterBand(col).ReadAsArray()
        values = []
        for value in bandValues:
            values.extend(value)
        out_csv[bandnames[col-1]] = values*n
    
    geotransform = geotiff.GetGeoTransform()
    # Loop through each pixel and extract coordinates and values
    x_geo, y_geo = [], []
    for y in range(geotiff.RasterYSize):
        for x in range(geotiff.RasterXSize):
            x_geo.append(geotransform[0] + (x+0.5) * geotransform[1] + y * geotransform[2])
            y_geo.append(geotransform[3] + x * geotransform[4] + (y+0.5) * geotransform[5])
    out_csv['x'] = x_geo*n
    out_csv['y'] = y_geo*n
    
    # from raster projection to 4269
    source_crs = pyproj.CRS.from_epsg(crs.split(":")[1])
    target_crs = pyproj.CRS.from_epsg(4269)
    latlonproj = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    coords = latlonproj.transform(out_csv['x'].values, out_csv['y'].values)
    out_csv['lon'] = coords[0]
    out_csv['lat'] = coords[1]
    geotiff = None
    return out_csv

#load ML GBM models
f = open('files/models.pckl', 'rb')
ghg_model, agb_model, bgb_model = pickle.load(f)
f.close()

#verify CRS or convert to WGS '84
shapefile.crs

# extract a single meadow and it's geometry bounds; buffer inwards by designated amount
meadowId = 5    # 9313 (crosses image boundary), 17902 (largest), 16658 (smallest)
feature = shapefile.loc[meadowId, ]
if feature.geometry.geom_type == 'Polygon':
    shapefile_bbox = ee.Geometry.Polygon(list(feature.geometry.exterior.coords)).buffer(-5)
elif feature.geometry.geom_type == 'MultiPolygon':
    shapefile_bbox = ee.Geometry.MultiPolygon(list(list(poly.exterior.coords) for poly in feature.geometry.geoms)).buffer(-5)

# convert landsat image collection over each meadow to list for iteration
landsat_images = landsat8_collection.filterBounds(shapefile_bbox).map(maskImage)
image_list = landsat_images.toList(landsat_images.size())
noImages = image_list.size().getInfo()

# clip flow, elevation and slope to meadow's bounds
flow_band = flow_acc.clip(shapefile_bbox)
dem_bands = dem.clip(shapefile_bbox)
slopeDem = slope.clip(shapefile_bbox)

# dataframe to store results for each meadow
cols = ['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2', 'Flow', 'Elevation', 'Slope', 'Precipitation',
        'Minimum_temperature', 'Maximum_temperature', 'SWE', 'X', 'Y', 'Longitude', 'Latitude', 'Date']
# confirm use of temperature and SWE above
all_data = pd.DataFrame(columns=cols)
mycrs = None
combined_image = None
var_col = []
bandnames = []
subregions = [shapefile_bbox]

# iterate through each landsat image
for idx in range(noImages):
    landsat_image = ee.Image(image_list.get(idx)).select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).toFloat()
    # use each date in which landsat image exists to extract bands of gridmet and daymet
    date = landsat_image.getInfo()['properties']['DATE_ACQUIRED']
    start_date = datetime.strptime(date, '%Y-%m-%d')
    
    gridmet_filtered = gridmet.filterDate(date, (start_date + timedelta(days=1)).strftime('%Y-%m-%d')).first().clip(shapefile_bbox)
    gridmet_30m = gridmet_filtered.resample('bilinear')
    daymetv4 = daymet.filterBounds(shapefile_bbox).filterDate(date, (start_date + timedelta(days=1)).strftime('%Y-%m-%d')).first()
    daymet_30m = daymetv4.resample('bilinear')
    
    # align other satellite data with landsat and make resolution uniform (30m)
    if idx == 0:     # only extract once for the same meadow
        mycrs = landsat_image.projection().getInfo()['crs']
        flow_30m = flow_band.resample('bilinear').toFloat()
        dem_30m = dem_bands.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear')
        slope_30m = slopeDem.reduceResolution(ee.Reducer.mean(), maxPixels=65536).resample('bilinear')
        combined_image = landsat_image.addBands(flow_30m).addBands(dem_30m).addBands(slope_30m).addBands(gridmet_30m).addBands(daymet_30m)
        bandnames = combined_image.bandNames().getInfo()
    else:
        combined_image = combined_image.addBands(landsat_image).addBands(gridmet_30m)
        bandnames = bandnames + bandnames[:9]     # 9 landsat and gridmet bands
    
    if feature.Area_km2 >= 100:     # split bounds of shapefile into 4 areas to stay within limit of image downloads
        coords = shapefile_bbox.bounds().coordinates().getInfo()[0]
        xmin, ymin = coords[0]
        xmax, ymax = coords[2]

        subregion_width = (xmax - xmin) / 2
        subregion_height = (ymax - ymin) / 2
        subregions = []
        for i in range(2):
            for j in range(2):
                subregion = ee.Geometry.Rectangle([xmin + i*subregion_width, ymin + j*subregion_height,
                                                   xmin + (i+1)*subregion_width, ymin + (j+1)*subregion_height])
                subregions.append(subregion.intersection(shapefile_bbox))

    for i, subregion in enumerate(subregions):
        temp_image = combined_image.clip(subregion)
        image_name = f'meadow_{meadowId}_{i}.tif'

        bandnames = temp_image.bandNames().getInfo()
        # suppress output of downloaded images
        with contextlib.redirect_stdout(None):
            geemap.ee_export_image(temp_image, filename=image_name, scale=30, region=shapefile_bbox, crs=mycrs)
        shapefile = None
        
        df = geotiffToCsv(image_name, bandnames, mycrs)
        nullIds =  list(np.where(df['tmmx'].isnull())[0])
        df.drop(nullIds, inplace = True)
        df.reset_index(drop=True, inplace=True)
        n = df.shape[0]
        
        # temporary dataframe for each iteration to be appended to the overall dataframe
        df['Date'] = [date]*n
        df.columns = cols
        df['Blue'] = [(x*2.75e-05 - 0.2) for x in df['Blue']]
        df['Green'] = [(x*2.75e-05 - 0.2) for x in df['Green']]
        df['Red'] = [(x*2.75e-05 - 0.2) for x in df['Red']]
        df['NIR'] = [(x*2.75e-05 - 0.2) for x in df['NIR']]
        df['SWIR_1'] = [(x*2.75e-05 - 0.2) for x in df['SWIR_1']]
        df['SWIR_2'] = [(x*2.75e-05 - 0.2) for x in df['SWIR_2']]
        df['NDVI'] = (df['NIR'] - df['Red'])/(df['NIR'] + df['Red'])
        df['NDWI'] = (df['Green'] - df['NIR'])/(df['Green'] + df['NIR'])
        df['EVI'] = 2.5*(df['NIR'] - df['Red'])/(df['NIR'] + 6*df['Red'] - 7.5*df['Blue'] + 1)
        df['SAVI'] = 1.5*(df['NIR'] - df['Red'])/(df['NIR'] + df['Red'] + 0.5)
        df['BSI'] = ((df['Red'] + df['SWIR_1']) - (df['NIR'] + df['Red']))/(df['Red'] + df['SWIR_1'] + df['NIR'] + df['Red'])
        
        var_col = [c for c in df.columns if c not in ['Date', 'X', 'Y', 'Longitude', 'Latitude']]
        df['CO2.umol.m2.s'] = ghg_model.predict(df.loc[:, var_col])
        all_data = pd.concat([all_data, df])
    print(idx, end=' ')

if not all_data.empty:
    # change dates from strings to actual dates and get indices of max NIR to predict AGB/BGB
    all_data['Date'] = pd.to_datetime(all_data['Date'], format="%Y-%m-%d")
    maxIds = all_data.groupby('Pixel')['NIR'].idxmax()
    all_data['HerbBio.g.m2'] = list(agb_model.predict(all_data.loc[maxIds, var_col]))*n
    all_data['Roots.kg.m2'] = list(bgb_model.predict(all_data.loc[maxIds, var_col]))*n

all_data.head()
# predict on dataframe
if not all_data.empty:
    # use projected coordinates so that resolution (30m) would be meaningful
    utm_lons, utm_lats = all_data['X'].values, all_data['Y'].values
    res = 30
    
    out_rasters = {1: ['_AGB.tif', 'HerbBio.g.m2'], 2: ['_BGB.tif', 'Roots.kg.m2'], 3: ['_GHG.tif', 'CO2.umol.m2.s']}
    for i in range(1,4):
        out_raster = "files/Image_meadow_" + str(round(feature.ID)) + out_rasters[i][0]
        # make geodataframe of relevant columns (exclude geometry column) and projected coordinates as crs; convert to raster
        gdf = gpd.GeoDataFrame(all_data.loc[:, var_col+[out_rasters[i][1]]], geometry=gpd.GeoSeries.from_xy(utm_lons, utm_lats),
                               crs=mycrs.split(":")[1])
        out_grd = make_geocube(vector_data=gdf, measurements=gdf.columns.tolist()[:-1], resolution=(-res, res))
        out_grd.rio.to_raster(out_raster)
