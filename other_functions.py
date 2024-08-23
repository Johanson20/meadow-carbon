# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:45:06 2024

@author: jonyegbula
"""

import os
import pandas as pd
import geopandas as gpd
import rioxarray as xr
import contextlib
import ee
import geemap
from shapely.geometry import Polygon
from osgeo import gdal, osr
from geocube.api.core import make_geocube


def GeotiffFromGEEImage(GEE_image, bounding_box, geotiffname, bandnames, epsg, res=30):
    '''
    Creates a single geotiff file of all "bandnames" in a constrained GEE image and plots a raster of each band. Supply constrained image like clipped flow accumulation, filtered gridmet/landsat, etc. and also the ee.Geometry region it was clipped to. Include epsg code of region, spatial resolution in meters, a list of valid bandnames in the image, as well as the raster's name
    '''
    myImage = GEE_image.select(bandnames)
    geemap.ee_export_image(myImage, filename=geotiffname, scale=res, region=bounding_box, crs='EPSG:'+epsg)
    geotiff = xr.open_rasterio(geotiffname)
    df = geotiff.to_dataframe(name='value').reset_index()
    df = df.pivot_table(index=['y', 'x'], columns='band', values='value').reset_index()
    geotiff.close()
    df.columns = ['Y', 'X'] + bandnames
    
    gdf = gpd.GeoDataFrame(df[bandnames], geometry=gpd.GeoSeries.from_xy(list(df['X']), list(df['Y'])), crs=epsg)
    out_grd = make_geocube(vector_data=gdf, measurements=bandnames, resolution=(-res, res))
    out_grd.rio.to_raster(f'{geotiffname.split(".tif")[0]}_out.tif')
    for band in bandnames:
        gdf.plot(column=band, cmap='viridis', legend=True)

# GeotiffFromGEEImage(gridmet.filterBounds(shapefile_bbox).first(), shapefile_bbox, "files/landsat.tif", ['SR_B2', 'SR_B5'], '32610')


def downloadBands(banded_image, image_name, region, mycrs, scale=30):
    '''
    Split an area (if it's too large to download singly) into 4 subareas and download the bands as a geotiff, then download subareas.
    '''
    with contextlib.redirect_stdout(None):
        geemap.ee_export_image(banded_image.clip(region), filename=image_name, scale=30, region=region, crs=mycrs)
    
    if os.path.exists(image_name):
        return
    else:
        xmin, ymin, xmax, ymax = region.geometry.bounds
        num_subregions = 2
        subregion_width = (xmax - xmin) / num_subregions
        subregion_height = (ymax - ymin) / num_subregions
        count = 0
        for i in range(num_subregions):
            for j in range(num_subregions):
                subarea = Polygon([(xmin + i*subregion_width, ymin + j*subregion_height),
                                   (xmin + (i+1)*subregion_width, ymin + j*subregion_height),
                                   (xmin + (i+1)*subregion_width, ymin + (j+1)*subregion_height),
                                   (xmin + i*subregion_width, ymin + (j+1)*subregion_height)])
                if subarea.intersects(region.geometry):
                    subregion = ee.Geometry.Rectangle(list(subarea.bounds)).intersection(region)
                    new_name = f'{image_name.split(".tif")[0]}_{count}.tif'
                    downloadBands(banded_image, new_name, subregion, mycrs, scale=30)

# downloadBands(combined_image, "meadow_2021_16696_4614_0.tif", shapefile_bbox, "EPSG:32610")


def geotiffToCsv(input_raster, csvfile, folderPath="", epsg=4269):
    '''
    Convert geotiff to raster and make coordinates in latlon from projected, supply epsg code of projection
    '''
    geotiff = gdal.Open(input_raster)
    nBands = geotiff.RasterCount
    out_csv = pd.DataFrame(columns=range(nBands))
    bandnames = []
    
    for band in range(1, 1+nBands):
        bandValues = geotiff.GetRasterBand(band).ReadAsArray()
        values = []
        for value in bandValues:
            values.extend(value)
        
        bandnames.append(geotiff.GetRasterBand(band).GetDescription())
        out_csv.iloc[:, band-1] = values
    out_csv.columns = bandnames
    
    geotransform = geotiff.GetGeoTransform()
    # Loop through each pixel and extract coordinates and values
    x_geo, y_geo = [], []
    for y in range(geotiff.RasterYSize):
        for x in range(geotiff.RasterXSize):
            x_geo.append(geotransform[0] + (x+0.5) * geotransform[1] + y * geotransform[2])
            y_geo.append(geotransform[3] + x * geotransform[4] + (y+0.5) * geotransform[5])
    out_csv['x'] = x_geo
    out_csv['y'] = y_geo
    
    # from raster projection to 4269
    mycrs = osr.SpatialReference()
    mycrs.ImportFromProj4(geotiff.GetProjection())
    target_crs = osr.SpatialReference()
    target_crs.ImportFromEPSG(epsg)  # WGS84
    transform = osr.CoordinateTransformation(mycrs, target_crs)
    points = [(x, y) for x, y in zip(x_geo, y_geo)]
    coords = transform.TransformPoints(points)
    out_csv['lon'] = [data[1] for data in coords]
    out_csv['lat'] = [data[0] for data in coords]
    geotiff = None
    out_csv.to_csv(csvfile, encoding='utf-8', index=False)

# geotiffToCsv("Treemap.tif", "out_xyz.csv", "", 32611)