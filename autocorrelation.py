# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:24:13 2024

@author: Johanson C. Onyegbula
"""


import numpy as np
import pandas as pd
from skgstat import Variogram
from pyproj import Transformer

data = pd.read_csv("Aboveground Biomass_RS Model.csv")
data.head()
data.columns

nullIds =  list(np.where(data['Latitude'].isnull())[0])    # null IDs
data.drop(nullIds, inplace = True)

z = data['HerbBio.g.m2']
lon, lat = data['Longitude'], data['Latitude']
transformer = Transformer.from_crs("EPSG:4269", "epsg:32611")
x, y = transformer.transform(lat, lon)

vg = Variogram(np.vstack((x,y)).T, z)
print(vg)
vg.plot()
vg.describe()

data['x'] = x
data['y'] = y
data.to_csv('data.csv', index=False)
