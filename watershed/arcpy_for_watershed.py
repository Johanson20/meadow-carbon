# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:34:12 2025

@author: Johanson C. Onyegbula
"""

import geopandas as gpd
import pandas as pd
import os
import warnings
import arcpy
# arcpy is accessible through any python IDLE/console under ArcGIS Pro in the start menu apps (after installation)

# Set working directory to folder containing GDB
mydir = "C:/Users/jonyegbula/Documents/ArcGIS/Projects/HydroDEM"
os.chdir(mydir)
warnings.filterwarnings("ignore")

epsg_crs = "EPSG:4326"
input_gdb = f"{mydir}/HydroDEM.gdb"
arcpy.env.workspace = input_gdb     # ensures arcpy accesses datasets directly from the geodatabase without path issues
arcpy.env.overwriteOutput = True    # allows overwriting of same datasets without failure if code is to be re-run
arcpy.ListRasters()
snapped_pts = gpd.read_file("../../../PointBlue/Code/files/snapped_cascade_pour_points.shp")
snapped_pts['ID'] = snapped_pts['ID'].astype('Int16')
cols = ["ID", "geometry", "MaxFlowAcc"]
all_data = pd.DataFrame(columns=cols)
Failed_Idx = []

# for each pour point, generate a unique watershed and note the failed indices
for idx in range(snapped_pts.shape[0]):
    point1 = snapped_pts.loc[[idx]]
    try:
        # write single pour point to a file and convert to a proper feature class
        point1.to_file("temp_file.shp")
        arcpy.conversion.FeatureClassToFeatureClass("temp_file.shp", input_gdb, "point1")
        # arcpy.ListFeatureClasses()    # list all feature classes/shapefiles
        
        # delineate single watershed, save it to re-read and convert the raster to a polygon
        watershed1 = arcpy.sa.Watershed("FlowDirection", "point1")
        watershed1.save("watershed1")
        arcpy.conversion.RasterToPolygon("watershed1", "vector_water.shp", "SIMPLIFY", "Value")
        # read single watershed polygon, modify attributes and append to the dataframe stack (all_data)
        wshed = gpd.read_file("vector_water.shp")
        wshed.drop(columns=["gridcode"], inplace=True)
        wshed['MaxFlowAcc'] = point1["Flow_accum"]
        wshed.columns = cols
        all_data = pd.concat([all_data, wshed])
    except:
        print(f"Meadow ID = {idx} didn't work!")
        Failed_Idx.append(idx)
    if idx%50 == 0: print(idx, end=' ')

# create watershed geodataframe and save as polygons to a file
watershed_gdf = gpd.GeoDataFrame({'Meadow_ID': list(all_data['ID']), 'MaxFlowAcc': list(all_data['MaxFlowAcc'])}, geometry=list(all_data['geometry']), crs=epsg_crs)
watershed_gdf.to_file("../../../PointBlue/Code/files/snapped_cascade_watershed_polygons.shp", driver="ESRI Shapefile")