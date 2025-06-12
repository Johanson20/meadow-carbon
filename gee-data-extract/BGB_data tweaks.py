# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:32:00 2025

@author: Johanson C. Onyegbula
"""

import os
import ee
import warnings
import pandas as pd

mydir = "Code"      # adjust directory
os.chdir(mydir)
warnings.filterwarnings("ignore")


# resample and reproject when image's pixel size is not 30m for both UTM zones
def resample10(image):
    return image.resample("bilinear").reproject(crs="EPSG:32610", scale=30)

def resample11(image):
    return image.resample("bilinear").reproject(crs="EPSG:32611", scale=30)


data = pd.read_csv("csv/Belowground Biomass_RS Model_Data.csv")
data.head()

# load relevant polaris datasets of UTM Zone 10 and extract each of the first unique 4 depths
perc_clay = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample10)
hydra_cond = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample10)
organic_m = ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').select("b1").map(resample10)
perc_sand = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').select("b1").map(resample10)

l1_perc_clay = ee.Image(perc_clay.toList(6).get(0))
l2_perc_clay = ee.Image(perc_clay.toList(6).get(1))
l3_perc_clay = ee.Image(perc_clay.toList(6).get(2))
l4_perc_clay = ee.Image(perc_clay.toList(6).get(3))
l1_perc_sand = ee.Image(perc_sand.toList(6).get(0))
l2_perc_sand = ee.Image(perc_sand.toList(6).get(1))
l3_perc_sand = ee.Image(perc_sand.toList(6).get(2))
l4_perc_sand = ee.Image(perc_sand.toList(6).get(3))
l1_hydra_cond = ee.Image(hydra_cond.toList(6).get(0))
l2_hydra_cond = ee.Image(hydra_cond.toList(6).get(1))
l3_hydra_cond = ee.Image(hydra_cond.toList(6).get(2))
l4_hydra_cond = ee.Image(hydra_cond.toList(6).get(3))
l1_organic_m = ee.Image(organic_m.toList(6).get(0))
l2_organic_m = ee.Image(organic_m.toList(6).get(1))
l3_organic_m = ee.Image(organic_m.toList(6).get(2))
l4_organic_m = ee.Image(organic_m.toList(6).get(3))

# load same as above for UTM Zone 11
perc_clay_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample11)
hydra_cond_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample11)
organic_m_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').select("b1").map(resample11)
perc_sand_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').select("b1").map(resample11)

l1_perc_clay_11 = ee.Image(perc_clay_11.toList(6).get(0))
l2_perc_clay_11 = ee.Image(perc_clay_11.toList(6).get(1))
l3_perc_clay_11 = ee.Image(perc_clay_11.toList(6).get(2))
l4_perc_clay_11 = ee.Image(perc_clay_11.toList(6).get(3))
l1_perc_sand_11 = ee.Image(perc_sand_11.toList(6).get(0))
l2_perc_sand_11 = ee.Image(perc_sand_11.toList(6).get(1))
l3_perc_sand_11 = ee.Image(perc_sand_11.toList(6).get(2))
l4_perc_sand_11 = ee.Image(perc_sand_11.toList(6).get(3))
l1_hydra_cond_11 = ee.Image(hydra_cond_11.toList(6).get(0))
l2_hydra_cond_11 = ee.Image(hydra_cond_11.toList(6).get(1))
l3_hydra_cond_11 = ee.Image(hydra_cond_11.toList(6).get(2))
l4_hydra_cond_11 = ee.Image(hydra_cond_11.toList(6).get(3))
l1_organic_m_11 = ee.Image(organic_m_11.toList(6).get(0))
l2_organic_m_11 = ee.Image(organic_m_11.toList(6).get(1))
l3_organic_m_11 = ee.Image(organic_m_11.toList(6).get(2))
l4_organic_m_11 = ee.Image(organic_m_11.toList(6).get(3))

L1_Clay, L2_Clay, L3_Clay, L4_Clay, L1_Hydra, L2_Hydra, L3_Hydra, L4_Hydra = [],[],[],[],[],[],[],[]
L1_Sand, L2_Sand, L3_Sand, L4_Sand, L1_Org, L2_Org, L3_Org, L4_Org = [],[],[],[],[],[],[],[]

# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    year, month, day = target_date.split("-")
    next_month = str(int(month)+1) if int(month) > 8 else "0" + str(int(month)%12+1)
    
    # compute values from daymetv4 (1km resolution) and gridmet/terraclimate (resolution of both is 4,638.3m)
    if x >= -120:   # corresponds to zone 32611
        l1_clay = l1_perc_clay_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l2_clay = l2_perc_clay_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l3_clay = l3_perc_clay_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l4_clay = l4_perc_clay_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l1_sand = l1_perc_sand_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l2_sand = l2_perc_sand_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l3_sand = l3_perc_sand_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l4_sand = l4_perc_sand_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l1_hydra = l1_hydra_cond_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l2_hydra = l2_hydra_cond_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l3_hydra = l3_hydra_cond_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l4_hydra = l4_hydra_cond_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l1_organic = l1_organic_m_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l2_organic = l2_organic_m_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l3_organic = l3_organic_m_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l4_organic = l4_organic_m_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    else:   # zone 32610
        l1_clay = l1_perc_clay.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l2_clay = l2_perc_clay.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l3_clay = l3_perc_clay.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l4_clay = l4_perc_clay.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l1_sand = l1_perc_sand.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l2_sand = l2_perc_sand.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l3_sand = l3_perc_sand.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l4_sand = l4_perc_sand.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l1_hydra = l1_hydra_cond.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l2_hydra = l2_hydra_cond.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l3_hydra = l3_hydra_cond.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l4_hydra = l4_hydra_cond.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l1_organic = l1_organic_m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l2_organic = l2_organic_m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l3_organic = l3_organic_m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        l4_organic = l4_organic_m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    
    L1_Clay.append(l1_clay)
    L2_Clay.append(l2_clay)
    L3_Clay.append(l3_clay)
    L4_Clay.append(l4_clay)
    L1_Hydra.append(l1_hydra)
    L2_Hydra.append(l2_hydra)
    L3_Hydra.append(l3_hydra)
    L4_Hydra.append(l4_hydra)
    L1_Sand.append(l1_sand)
    L2_Sand.append(l2_sand)
    L3_Sand.append(l3_sand)
    L4_Sand.append(l4_sand)
    L1_Org.append(l1_organic)
    L2_Org.append(l2_organic)
    L3_Org.append(l3_organic)
    L4_Org.append(l4_organic)
    
    if idx%50 == 0: print(idx, end=' ')

data['L1_Clay'] = L1_Clay
data['L2_Clay'] = L2_Clay
data['L3_Clay'] = L3_Clay
data['L4_Clay'] = L4_Clay
data['L1_Sand'] = L1_Sand
data['L2_Sand'] = L2_Sand
data['L3_Sand'] = L3_Sand
data['L4_Sand'] = L4_Sand
data['L1_Hydra_Conduc'] = L1_Hydra
data['L2_Hydra_Conduc'] = L2_Hydra
data['L3_Hydra_Conduc'] = L3_Hydra
data['L4_Hydra_Conduc'] = L4_Hydra
data['L1_Organic_Matter'] = L1_Org
data['L2_Organic_Matter'] = L2_Org
data['L3_Organic_Matter'] = L3_Org
data['L4_Organic_Matter'] = L4_Org

data.to_csv("files/Belowground Biomass_RS Model_Data.csv", index=False)


# Another extraction (depth classes of polaris soil averaged for first 3; 4th depth used alone)
perc_clay = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample10)
hydra_cond = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample10)
organic_m = ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').select("b1").map(resample10)

shallow_perc_clay = ee.ImageCollection(perc_clay.toList(3)).mean()
deep_perc_clay = ee.Image(perc_clay.toList(6).get(3))
shallow_hydra_cond = ee.ImageCollection(hydra_cond.toList(3)).mean()
deep_hydra_cond = ee.Image(hydra_cond.toList(6).get(3))
shallow_organic_m = ee.ImageCollection(organic_m.toList(3)).mean()

lithology = ee.Image("CSP/ERGo/1_0/US/lithology").select("b1").resample("bilinear").reproject(crs="EPSG:32610", scale=30)
topo_index = ee.Image("CSP/ERGo/1_0/US/topoDiversity").select("constant").resample("bilinear").reproject(crs="EPSG:32610", scale=30)

perc_clay_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').select("b1").map(resample11)
hydra_cond_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').select("b1").map(resample11)
organic_m_11 = ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').select("b1").map(resample11)

shallow_perc_clay_11 = ee.ImageCollection(perc_clay_11.toList(3)).mean()
deep_perc_clay_11 = ee.Image(perc_clay_11.toList(6).get(3))
shallow_hydra_cond_11 = ee.ImageCollection(hydra_cond_11.toList(3)).mean()
deep_hydra_cond_11 = ee.Image(hydra_cond_11.toList(6).get(3))
shallow_organic_m_11 = ee.ImageCollection(organic_m_11.toList(3)).mean()

lithology_11 = ee.Image("CSP/ERGo/1_0/US/lithology").select("b1").resample("bilinear").reproject(crs="EPSG:32611", scale=30)
topo_index_11 = ee.Image("CSP/ERGo/1_0/US/topoDiversity").select("constant").resample("bilinear").reproject(crs="EPSG:32611", scale=30)

Shallow_Clay, Shallow_Hydra, Organic, Deep_Clay, Deep_Hydra = [], [], [], [], []

# populate bands by applying above functions for each pixel in dataframe
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    year, month, day = target_date.split("-")
    next_month = str(int(month)+1) if int(month) > 8 else "0" + str(int(month)%12+1)
    
    # compute values from daymetv4 (1km resolution) and gridmet/terraclimate (resolution of both is 4,638.3m)
    if x >= -120:   # zone 32611
        shallow_clay = shallow_perc_clay_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        deep_clay = deep_perc_clay_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        shallow_hydra = shallow_hydra_cond_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        deep_hydra = deep_hydra_cond_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        organic = shallow_organic_m_11.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    else:   # zone 32610
        shallow_clay = shallow_perc_clay.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        deep_clay = deep_perc_clay.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        shallow_hydra = shallow_hydra_cond.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        deep_hydra = deep_hydra_cond.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
        organic = shallow_organic_m.reduceRegion(ee.Reducer.mean(), point, 30).getInfo()['b1']
    
    Shallow_Clay.append(shallow_clay)
    Shallow_Hydra.append(shallow_hydra)
    Organic.append(organic)
    Deep_Clay.append(deep_clay)
    Deep_Hydra.append(deep_hydra)
    
    if idx%50 == 0: print(idx, end=' ')

data['Shallow_Clay'] = Shallow_Clay
data['Deep_Clay'] = Deep_Clay
data['Shallow_Hydra_Conduc'] = Shallow_Hydra
data['Deep_Hydra_Conduc'] = Deep_Hydra
data['Organic_Matter'] = Organic

data.to_csv("files/Belowground Biomass_RS Model_Data.csv", index=False)