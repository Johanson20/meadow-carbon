# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:25:44 2024

@author: jonyegbula
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from affine import Affine
from pysheds.grid import Grid
from shapely.geometry import LineString, MultiPolygon

mydir = "C:/Users/jonyegbula/Documents/PointBlue/Code"
os.chdir(mydir)

# Function to map geographic coordinates to grid indices
def map_to_pixels(x, y, grid_bounds, cellsize):
    col = int((x - grid_bounds[0]) / cellsize)
    row = int((grid_bounds[3] - y) / cellsize)
    return row, col

# read in shapefile, the hydroshed DEM and adjust flats and depressions
epsg_crs = "EPSG:4326"
shapefile = gpd.read_file("files/AllPossibleMeadows_2024-11-5.shp").to_crs(epsg_crs)
grid = Grid.from_raster('files/merged_dem.tif')
dem = grid.read_raster('files/merged_dem.tif')
flooded_dem = grid.fill_depressions(dem)
inflated_dem = grid.resolve_flats(flooded_dem)

# transformation parameters of DEM for use
transform = dem.affine
cellsize = abs(transform[0])    # (90m resolution approximately)

# compute flow direction
d8 = (1, 2, 3, 4, 5, 6, 7, 8)
fdir = grid.flowdir(inflated_dem, dirmap=d8)

# Delineate watershed using flow direction
acc = grid.accumulation(fdir, dirmap=d8)


def watershedValues(meadowIdx):
    # extract boundary coordinates of meadow
    feature = shapefile.loc[meadowIdx, :].geometry
    meadowId = int(shapefile.loc[meadowIdx, :].ID)
    if feature.geom_type == 'Polygon':
        meadow_bounds = feature.bounds
        boundary = LineString(list(feature.exterior.coords))
    elif feature.geom_type == 'MultiPolygon':
        boundary = LineString([coord for polygon in feature.geoms for coord in polygon.exterior.coords])
        meadow_bounds = MultiPolygon([polygon for polygon in feature.geoms]).bounds
    boundary_coords = np.array(boundary.coords)
    min_x, min_y, max_x, max_y = meadow_bounds
    
    try:
        min_row, max_col = ~transform * (min_x, min_y)  # Upper-left corner
        max_row, min_col = ~transform * (max_x, max_y)  # Bottom-right corner
        
        # Convert to integer indices
        min_row, max_col = int(np.floor(min_row)), int(np.ceil(max_col))
        max_row, min_col = int(np.ceil(max_row)), int(np.floor(min_col))
        
        # Clip the flow accumulation DEM using these indices and calculate transformation parameters
        clipped_dem = acc[min_row:max_row, min_col:max_col]
        new_affine = transform * Affine.translation(min_row, min_col)
        nrows, ncols = clipped_dem.shape
        grid_bounds = (new_affine[2], new_affine[5] + nrows * new_affine[4], new_affine[2] + ncols * new_affine[0], new_affine[5])
    
        if len(set(clipped_dem.flatten())) > 1:     # ensure the values are different so there is a maximum flow cell
            max_acc_cell = np.where(clipped_dem == clipped_dem.max())
            # pour point is cell of highest flow accumulation
            x_pour_point, y_pour_point = new_affine * (max_acc_cell[1][0], max_acc_cell[0][0])
        else:
            # Find elevation at each boundary point
            boundary_elevations = []
            for x, y in boundary_coords:
                try:
                    row, col = map_to_pixels(x, y, grid_bounds, cellsize)
                    elevation = clipped_dem[row, col]
                    boundary_elevations.append(elevation)
                except IndexError:       # Handle cases where coordinates are out of grid bounds
                    boundary_elevations.append(np.nan)
            
            # pour point should be lowest elevation in meadow
            min_index = np.argmin(boundary_elevations)
            x_pour_point, y_pour_point = boundary_coords[min_index]
        
        
        # Convert pour point to grid indices
        lon, lat = x_pour_point, y_pour_point
        row, col = (int((new_affine[5] - lat) / cellsize), int((lon - new_affine[2]) / cellsize))
        
        # calculate upland accumulated area in units of cell size
        upland_area = clipped_dem[row, col] * (cellsize**2)
        
        # Calculate slope at pour point (units need adjustment based on cell size)
        dz_dx, dz_dy = np.gradient(inflated_dem[min_row:max_row, min_col:max_col], cellsize)
        slope_at_pour = np.sqrt(dz_dx[row, col]**2 + dz_dy[row, col]**2)
        
        return [meadowId, lon, lat, upland_area, slope_at_pour]
    except:
        return [meadowId, np.nan, np.nan, np.nan, np.nan]


watersheds = pd.DataFrame(columns=['ID', 'Longitude', 'Latitude', 'Upland_Area', 'Slope_at_Pour'])

for meadowIdx in shapefile.index:
    watersheds.loc[meadowIdx, :] = watershedValues(meadowIdx)

watersheds.to_csv("csv/watersheds.csv", index=False)

# upland_area * 90**2     # Probable area in square meters
# slope_at_pour/(90**2)   # Probable slope in degrees
