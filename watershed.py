# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:25:44 2024

@author: jonyegbula
"""

import os
import geopandas as gpd
import numpy as np
from pysheds.grid import Grid
from shapely.geometry import LineString

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
grid = Grid.from_raster('files/n30w120_con.tif')
dem = grid.read_raster('files/n30w120_con.tif')
flooded_dem = grid.fill_depressions(dem)
inflated_dem = grid.resolve_flats(flooded_dem)

# transformation parameters of DEM for use
transform = dem.affine
nrows, ncols = dem.shape

# Compute grid bounds (xmin, ymin, xmax, ymax)
grid_bounds = (transform[2], transform[5] + nrows * transform[4], transform[2] + ncols * transform[0], transform[5])
cellsize = abs(transform[0])    # (90m resolution approximately)

# compute flow direction
d8 = (1, 2, 3, 4, 5, 6, 7, 8)
fdir = grid.flowdir(inflated_dem, dirmap=d8)

# Delineate watershed using flow direction
acc = grid.accumulation(fdir, dirmap=d8)


def watershedValues(meadowIdx):
    # extract boundary coordinates of meadow
    feature = shapefile.loc[meadowIdx, :].geometry
    if feature.geom_type == 'Polygon':
        boundary = LineString(list(feature.exterior.coords))
    elif feature.geom_type == 'MultiPolygon':
        boundary = LineString([coord for polygon in feature.geoms for coord in polygon.exterior.coords])
    boundary_coords = np.array(boundary.coords)
    
    # Find elevation at each boundary point
    boundary_elevations = []
    for x, y in boundary_coords:
        try:
            row, col = map_to_pixels(x, y, grid_bounds, cellsize)
            elevation = inflated_dem[row, col]
            boundary_elevations.append(elevation)
        except IndexError:       # Handle cases where coordinates are out of grid bounds
            boundary_elevations.append(np.nan)
    
    # pour point should be lowest elevation in meadow
    min_index = np.argmin(boundary_elevations)
    pour_point = boundary_coords[min_index]
    
    # Convert pour point to grid indices
    lon, lat = pour_point
    row, col = (int((transform[5] - lat) / cellsize), int((lon - transform[2]) / cellsize))
    
    # calculate upland accumulated area in units of cell size
    upland_area = acc[row, col] * (cellsize**2)
    
    # Calculate slope at pour point (units need adjustment based on cell size)
    # Use central differences to approximate slope (rise/run)
    dz_dx, dz_dy = np.gradient(inflated_dem, cellsize)
    slope_at_pour = np.sqrt(dz_dx[row, col]**2 + dz_dy[row, col]**2)
    # slope_at_pour/(90**2)
    
    return upland_area, slope_at_pour


upland_area, slope_at_pour = watershedValues(16461)

upland_area * 90**2     # Probable area in square meters
slope_at_pour/(90**2)   # Probable slope in degrees
