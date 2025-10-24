# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""

import os, ee
import geopandas as gpd
import math
os.chdir("Code")    # change path to directory of folder

ee.Initialize()
landsat8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate('2022-07-01', '2022-07-31')

shapefile = gpd.read_file("files/AllPossibleMeadows_2025-10-22.shp")

import geemap
import ipyleaflet
import ipywidgets as widgets


def zoom_level(area):
    ''' zoom_level ranges from 10 (largest of 369.83082 km2) to 19 (smallest of 0.0007 km2) for all polygons
     Each zoom-out approximately quadruples the area viewed (hence log 2)
     calculate deviation or zoom-out extent from 19'''
    tradeoff = math.log2(area/0.0007)
    return (19 - round(tradeoff/2))


# change button value of selection to one of the options
def handle_selection(change):
    global selection
    if change['name'] == 'value' and change['new'] is not None:
        selection = True

ids = shapefile.ID
selection = False
toDel = []
toEdit = []

for i in ids[:3]:   # modify to all IDs ("ids")
    # extract centroid coordinates of one polygon
    row = shapefile[shapefile.ID == i].iloc[0]
    feature = row.geometry
    lon, lat = feature.centroid.coords[0]
    
    # create visualization layer to reasonable zoom level based on polygon size
    Map = geemap.Map(center=[lat, lon], zoom=zoom_level(row.Area_km2))
    Map.add_basemap("SATELLITE")
    
    # display polygon on satellite layer and options for action to be done to polygon
    gdf_selected = gpd.GeoDataFrame(geometry=[feature])
    geo_data = ipyleaflet.GeoData(geo_dataframe=gdf_selected, style={'color': 'red', 'fillOpacity':0.01})
    dropdown = widgets.Dropdown(options=["--- Select One ---", "Preserve", "Delete", "Edit later"], 
                               value="--- Select One ---", description="Take action:")
    dropdown.observe(handle_selection, names="value")
    output_control = ipyleaflet.WidgetControl(widget=dropdown, position="topright")
    
    Map.add_layer(geo_data)
    Map.add_control(output_control)
    display(Map)
    
    # waits until an option is selected
    while not selection:
        pass
    selection = False
    
    # take necessary action based on selected option before moving to next polygon in iteration
    if dropdown.value == "Delete":
        toDel.append(i)
    elif dropdown.value == "Edit later":
        toEdit.append(i)
    print('done')
    Map.close()

# print what polygons were chosen as real meadows to be preserved, edited or discarded (false meadows)
print(toEdit)
print(toDel)
print(dropdown.value)
