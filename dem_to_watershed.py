# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:44:34 2025

@author: jonyegbula
"""

import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.features import geometry_mask
import glob
import geopandas as gpd
import numpy as np
from pysheds.grid import Grid
import os
import fiona
from shapely.geometry import Point, Polygon

mydir = "R:/SCRATCH/jonyegbula/meadow-carbon"
os.chdir(mydir)

epsg_crs = "EPSG:4326"
meadows = gpd.read_file("files/AllPossibleMeadows_2025-03-07.shp").to_crs(epsg_crs)

# Specify the folder containing raster files and open them as datasets
raster_files = glob.glob("files/usgs_10m*.tif")
src_files_to_mosaic = [rasterio.open(fp) for fp in raster_files]
# extract metadata from one of the source files
out_meta = src_files_to_mosaic[0].meta.copy()

# mosaic files together nad update metadata
mosaic, out_trans = merge(src_files_to_mosaic, method='max')
out_meta.update({"driver": "GTiff", "nodata": -9999, "height": mosaic.shape[1],
    "width": mosaic.shape[2], "transform": out_trans})
# Save the output raster (optional)
with rasterio.open("files/sierra_nevada_10m.tif", "w", **out_meta) as dest:
    dest.write(mosaic)
del mosaic, src_files_to_mosaic, out_trans   # save space

# extract geometry of meadow for clipping
geoms = [geom.__geo_interface__ for geom in meadows.geometry]
with rasterio.open("files/sierra_nevada_10m.tif") as src:
    clipped_image, clipped_transform = mask(src, geoms, crop=True)
out_meta.update({"driver": "GTiff", "height": clipped_image.shape[1],
    "width": clipped_image.shape[2], "transform": clipped_transform})
# Save the output raster
with rasterio.open("files/cascade_nevada_10m.tif", "w", **out_meta) as dest:
    dest.write(clipped_image)
del clipped_image    # save space

grid = Grid.from_raster('files/sierra_nevada_10m.tif')
dem = grid.read_raster('files/sierra_nevada_10m.tif')
flooded_dem = grid.fill_depressions(dem)
inflated_dem = grid.resolve_flats(flooded_dem)

fdir = Grid.from_raster('files/FlowDirectionSierra.tif')
fdir = fdir.read_raster('files/FlowDirectionSierra.tif')
acc = Grid.from_raster('files/FlowAccumulationSierra.tif')
acc = acc.read_raster('files/FlowAccumulationSierra.tif')
transform = acc.affine

# compute flow direction and accumulation
d8 = (1, 2, 4, 8, 16, 32, 64, 128)
fdir = grid.flowdir(inflated_dem, dirmap=d8)
acc = grid.accumulation(fdir, dirmap=d8)
out_meta.update(dtype=rasterio.float32, count=1, nodata=-9999, height=acc.shape[0],
    width=acc.shape[1], transform=transform)  # update metadata
del flooded_dem, inflated_dem

# save rasters (optional)
with rasterio.open("files/flow_direction.tif", "w", **out_meta) as dest:
    dest.write(fdir, 1)
with rasterio.open("files/flow_accumulation.tif", "w", **out_meta) as dest:
    dest.write(acc, 1)

# convert meadows to raster mask and apply to flow accumulation for each meadow
meadow_mask = geometry_mask(meadows.geometry, acc.shape, transform, invert=True)
acc_masked = np.where(meadow_mask, acc, np.nan)

# convert flow accumulation mask to points
points = []
for row in range(acc_masked.shape[0]):
    for col in range(acc_masked.shape[1]):
        if not np.isnan(acc_masked[row, col]):
            x, y = transform*(col, row)
            points.append(Point(x, y))

# create geodataframe of points and extract flow accumulation values to them
points_gdf = gpd.GeoDataFrame(geometry=points, crs=epsg_crs)
points_gdf['Flow_accumulation'] = acc_masked[~np.isnan(acc_masked)]
points_gdf.head()
del meadow_mask, acc_masked

# spatially join points to meadows, to extract max flow accumulation per ID
pts_joined = gpd.sjoin(points_gdf, meadows, how="left", predicate="within")
pts_joined.dropna(subset=['ID', 'Flow_accumulation'], inplace=True)
pts_joined.reset_index(drop=True, inplace=True)
pts_joined.head()
pts_joined.columns
pts_joined.drop(columns=list(pts_joined.columns)[7:-1] + ['index_right'], axis=1, inplace=True)
len(set(pts_joined.ID))
pts = pts_joined.loc[pts_joined.groupby("ID")['Flow_accumulation'].idxmax()]
pts.reset_index(drop=True, inplace=True)
pts.Flow_accumulation = pts.Flow_accumulation/1000  # division due to data limuts of values
pts.head()
# save the pour points
with fiona.Env(SHAPE_RESTORE_SHX="Yes"):
    pts.to_file("files/sierra_pour_points.shp", driver="ESRI Shapefile")
del points, points_gdf, pts_joined

# create watershed from pour point row-col coordinates and flow direction
IDs, X_coords, Y_coords, Max_Flow_Acc = [], [], [], []
watersheds, water_poly, validPoly = [], [], []
for idx in range(pts.shape[0]):
    row = pts.loc[idx, :]
    # append other variables
    IDs.append(row.ID)
    x_pour_point, y_pour_point = row.geometry.centroid.x, row.geometry.centroid.y
    X_coords.append(x_pour_point)
    Y_coords.append(y_pour_point)
    Max_Flow_Acc.append(row.Flow_accum)
    # delineate unique watershed and convert to polygon (after including grid/raster)
    watershed = grid.catchment(x_pour_point, y_pour_point, fdir, dirmap=d8, xytype='coordinate')
    # watersheds.append(watershed)
    mask = watershed.astype(bool)
    rows, cols = np.where(mask)
    coords = [transform*(c,r) for r, c in zip(rows, cols)]
    if len(coords) > 2:
        water_poly.append(Polygon(coords))
        validPoly.append(idx)
    if idx%50 == 0: print(idx, end=' ')

# create the watersheds polygons file and save
watershed_gdf = gpd.GeoDataFrame({'Meadow_ID': validPoly, 'MaxFlowAcc': [Max_Flow_Acc[i] for i in validPoly], 'X_Pour_Pt': [X_coords[i] for i in validPoly],
                                  'Y_Pour_Pt': [Y_coords[i] for i in validPoly]}, geometry=water_poly, crs=epsg_crs)
with fiona.Env(SHAPE_RESTORE_SHX="Yes"):
    watershed_gdf.to_file("files/cascade_watershed_polygons.shp", driver="ESRI Shapefile")

# save the watershed raster
watershed_raster = np.zeros_like(fdir)
for ws in watersheds:
    watershed_raster[ws > 0] = ws[ws > 0]
with rasterio.open("files/watersheds.tif", "w", driver="GTiff", height=watershed_raster.shape[0], 
                   width=watershed_raster.shape[1], count=1, dtype=watershed_raster.dtype, 
                   crs=epsg_crs, transform=transform) as dst:
    dst.write(watershed_raster, 1)
