# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:17:24 2025

@author: jonyegbula
"""

import glob, ee
ee.Initialize()

# delete assets for a particular year
all_assets = ee.data.listAssets({"parent": "projects/ee-jonyegbula/assets"})["assets"]
assets_2007 = [a['name'] for a in all_assets if '2024' in a['name']]
for asset in assets_2007: ee.data.deleteAsset(asset)

# set visibility of all assets
acl_update = {'all_users_can_read': True}
asset = "projects/ee-jonyegbula/assets/"
for year in range(1984, 2025):
    tifs = glob.glob(f"files/results/{year}*.tif")
    for tif in tifs:
        asset_id = asset + tif[14:-4]
        ee.data.setAssetAcl(asset_id, acl_update)
    print(year, end=' ')

# get max 98th percentile value for each attribute for GEE app
store = [0,0,0,0,0,0, 0,0,0,0,0,0]
for year in range(1986, 2025):
    data = pd.read_csv(f"files/results/{year}_Meadows.csv")
    for i in range(12):
        col = data.columns[i+2]
        store[i] = max(store[i], data[col].quantile(0.98))
    print(year, end=' ')
store

data[['ModisNPP', 'LandsatNPP']] *= 0.1

data.loc[data['HerbBio.g.m2'] < 0, 'HerbBio.g.m2'] = 0
data['ANPP'] = data['HerbBio.g.m2']*0.433

data.loc[data['Roots.kg.m2'] < 0, 'Roots.kg.m2'] = 0
data['Root_Turnover'] = (data['Roots.kg.m2']*0.49 - ((data['Roots.kg.m2']*0.49)*np.exp(-0.53)))*0.368*1000
data['Root_Exudates'] = data['Roots.kg.m2']*1000*data['Active_growth_days']*12*1.04e-4
data['BNPP'] = data['Root_Turnover'] + data['Root_Exudates']

data.head()

data = pd.read_csv("files/results/2024_Meadows.csv")
jepson = shapefile[shapefile.ID == meadowId].EcoRegion.values[0]


for year in range(1984, 2025):
    outfile = f'files/results/{year}_Meadows.csv'
    statsfile = f'files/results/{year}_Meadows_stats.csv'
    stats = pd.read_csv(statsfile)
    stats['ID'], stats['PixelCount'] = [int(x) for x in stats['ID']], [int(x) for x in stats['PixelCount']]
    allID, allJep = [], []
    all_data = pd.read_csv(outfile)
    for idx in range(stats.shape[0]):
        meadowId, ndim, jepson = stats.iloc[idx, [0,1,-1]]
        ndim = stats[stats.ID == meadowId].PixelCount.values[0]
        allID.extend([meadowId]*ndim)
        allJep.extend([jepson]*ndim)
    all_data['ID'], all_data['Jepson_Region'] = allID, allJep
    all_data['NEP_Cap_1000'] = [val if val <= 1000 else 1000 for val in all_data['NEP']]
    all_data.to_csv(outfile, index=False)
    print(year, end=" ")


print("\nTRAINING DATA:\nRoot Mean Squared Error (RMSE) = {:.4f}".format(train_rmse))
print("Correlation coefficient (R) = {:.4f}".format(train_corr[0][1]))
print("\nTEST DATA:\nRoot Mean Squared Error (RMSE) = {:.4f}".format(test_rmse, test_mae))
print("Correlation coefficient (R) = {:.4f}".format(test_corr[0][1]))
print("\nMean Training Percentage Bias = {:.4f} %\nMean Test Percentage Bias = {:.4f} %".format(train_p_bias, test_p_bias))


with PdfPages('files/Largest.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(column=response_col, cmap=ListedColormap(['#FF0000', '#FFFF00', '#00FFFF', '#0000FF']), legend=True, ax=ax)
    pdf.savefig(fig)
    plt.close(fig)


data = pd.read_csv("csv/Aboveground Biomass_RS Model_5_year_Data.csv")
BNPPs, ANPPs = [], []
for idx in range(data.shape[0]):
    # extract coordinates and date from csv
    x, y = data.loc[idx, ['Longitude', 'Latitude']]
    point = ee.Geometry.Point(x, y)
    target_date = data.loc[idx, 'SampleDate']
    year, month, day = target_date.split("-")
    df = pd.read_csv(f"files/results/{year}_Meadows.csv")
    # next_month = str(int(month)+1) if int(month) > 8 else "0" + str(int(month)%12+1)
    prev_5_year = str(int(year)-6) + "-10-01"
    
    landsat = landsat_collection.filterBounds(point).filterDate(prev_5_year, year+"-10-01")
    bands_June, bands_Sept, utm, integrals, snow_days, wet_days, growth_days = extractAllValues(landsat, year)
    
    if not bands_June['Blue'] or not bands_Sept['Blue']:     # drop rows that returned null band values
        data.drop(idx, inplace=True)
        print("Row", idx, "dropped!")
        continue

for year in range(1984, 2025):
    # df = pd.read_csv(f"files/results/{year}_Meadows.csv")
    data = pd.read_csv(f"files/{year}_Meadows_stats.csv")
    # data['ID'] = data['ID'].astype(int)
    # data = data.sort_values(by='ID', ascending=True)
    # data['NEP_Cap_1000_mean'] = list(df.groupby('ID')['NEP_Cap_1000'].mean())
    # data['NEP_sum'] = list(df.groupby('ID')['NEP'].sum())
    # data['NEP_Cap_1000_sum'] = list(df.groupby('ID')['NEP_Cap_1000'].sum())
    data[['NEP_Cap_1000_sum', 'NEP_sum']] *= 900
    data.to_csv(f"files/{year}_Meadows_stats.csv", index=False)
    print(year, end=' ')
    x = len([x for x in df['NEP'] if x >= 1000])
    val = x/df.shape[0]*100
    x2 = len([x for x in data['NEP_mean'] if x >= 1000])
    val2 = x2/data.shape[0]*100
    print(f"{year}:\tNEP (all pixels) >= 1000 = {val:.2f}%;\t\tMean NEP (for {year}) >= 1000 = {val2:.2f}%")


from shapely.geometry import Point
data = pd.read_csv("csv/Belowground Biomass_RS Model_5_year_Data.csv")
data["geometry"] = data.apply(lambda r: Point(r.Longitude, r.Latitude), axis=1)
data = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")
all_data = gpd.sjoin(data, shapefile[["ID","geometry"]], how="left")
all_data.columns = list(all_data.columns[:-1]) + ['ID']

def get_closest_nep(csv_path, meadow_id, lon, lat, chunksize=200000):
    best_row = None
    best_dist = float("inf")
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        sub = chunk[chunk["ID"] == meadow_id]
        if sub.empty:
            continue
        # distance to each pixel in chunk
        d = np.sqrt((sub["X"] - lon)**2 + (sub["Y"] - lat)**2)
        idx = d.idxmin()
        if d.loc[idx] < best_dist:
            best_row = sub.loc[idx]
            best_dist = d.loc[idx]
    return [best_row["ANPP"], best_row["BNPP"], best_row["Rh"], best_row["NEP"]] if best_row is not None else None

results = []
for _, r in all_data.iterrows():
    nep = get_closest_nep(f"files/results/{year}_Meadows.csv", r.ID, r.Longitude, r.Latitude)
    results.append(nep)
    if _%20 == 0: print(_, end=' ')

data["Model_ANPP"] = x["Model_ANPP"]
data["Model_BNPP"] = x["Model_BNPP"]
data["Model_Rh"] = x["Model_Rh"]
data["Model_NEP"] = x["Model_NEP"]
data.drop("geometry", axis=1, inplace=True)
data.to_csv("csv/Belowground Biomass_RS Model_NPP.csv", index=False)

data = pd.read_csv("csv/Belowground Biomass_RS Model_NPP.csv")
data['Model_NPP'] = data['Model_ANPP'] + data['Model_BNPP']


data = pd.read_csv("csv/Belowground Biomass_RS Model_NPP.csv")
data.dropna(subset=['BNPP', 'LandsatNPP', 'ModisNPP'], inplace=True)
with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(10, 10))
    regressor = LinearRegression()
    test_y = np.array(data['BNPP']).reshape(-1,1)
    test_pred_y = np.array(data['LandsatNPP']).reshape(-1,1)
    regressor.fit(test_y, test_pred_y)
    y_pred = regressor.predict(test_y)
    ax.scatter(data['BNPP'], data['LandsatNPP'], color='blue', label='Landsat NPP CONUS')
    ax.scatter(data['BNPP'], data['ModisNPP'], color='orange', label='MODIS NPP CONUS')
    ax.plot(data['BNPP'], y_pred, color='blue', label='Regression line of Landsat NPP')
    test_pred_y = np.array(data['ModisNPP']).reshape(-1,1)
    regressor.fit(test_y, test_pred_y)
    y_pred = regressor.predict(test_y)
    ax.plot(data['BNPP'], y_pred, color='orange', label='Regression line of MODIS NPP')
    ax.plot(data['BNPP'], data['BNPP'], linestyle='dotted', color='black', label='1:1 line')
    ax.set_xlabel("BNPP of Training Samples (g C m-2)", fontweight='bold', fontsize=16)
    ax.set_ylabel("NPP CONUS (g C m-2)", fontweight='bold', fontsize=16)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    ax.set_title("Scatter Plot of Training sample BNPP and Satellite Estimated NPP (Net Primary Production)", fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig("files/BGB_NPP.png", dpi=300)
    plt.close(fig)
data = pd.read_csv("csv/Aboveground Biomass_RS Model_NPP.csv")
data.dropna(subset=['ANPP', 'LandsatNPP', 'ModisNPP'], inplace=True)
with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(10, 10))
    regressor = LinearRegression()
    test_y = np.array(data['ANPP']).reshape(-1,1)
    test_pred_y = np.array(data['LandsatNPP']).reshape(-1,1)
    regressor.fit(test_y, test_pred_y)
    y_pred = regressor.predict(test_y)
    ax.scatter(data['ANPP'], data['LandsatNPP'], color='blue', label='Landsat NPP CONUS')
    ax.scatter(data['ANPP'], data['ModisNPP'], color='orange', label='MODIS NPP CONUS')
    ax.plot(data['ANPP'], y_pred, color='blue', label='Regression line of Landsat NPP')
    test_pred_y = np.array(data['ModisNPP']).reshape(-1,1)
    regressor.fit(test_y, test_pred_y)
    y_pred = regressor.predict(test_y)
    ax.plot(data['ANPP'], y_pred, color='orange', label='Regression line of MODIS NPP')
    ax.plot(data['ANPP'], data['ANPP'], linestyle='dotted', color='black', label='1:1 line')
    ax.set_xlabel("ANPP of Training Samples (g C m-2)", fontweight='bold', fontsize=16)
    ax.set_ylabel("NPP CONUS (g C m-2)", fontweight='bold', fontsize=16)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    ax.set_title("Scatter Plot of Training sample ANPP and Satellite Estimated NPP (Net Primary Production)", fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig("files/AGB_NPP.png", dpi=300)
    plt.close(fig)


data = pd.read_csv("csv/Belowground Biomass_RS Model_NPP.csv")
data1 = pd.read_csv("csv/Aboveground Biomass_RS Model_NPP.csv")
data['Year'] = [x[:4] for x in data['SampleDate']]
data1['Year'] = [x[:4] for x in data1['SampleDate']]
data2 = pd.merge(data, data1, on=['ID', 'Year'])
data2['NPP'] = data2['ANPP'] + data2['BNPP']
data2.dropna(subset=['NPP', 'LandsatNPP_x', 'ModisNPP_x'], inplace=True)
with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(4, 4))
    regressor = LinearRegression()
    test_y = np.array(data2['NPP']).reshape(-1,1)
    test_pred_y = np.array(data2['LandsatNPP_x']).reshape(-1,1)
    regressor.fit(test_y, test_pred_y)
    y_pred = regressor.predict(test_y)
    ax.scatter(data2['NPP'], data2['LandsatNPP_x'], color='blue', label='Landsat NPP CONUS')
    ax.scatter(data2['NPP'], data2['ModisNPP_x'], color='orange', label='MODIS NPP CONUS')
    plt.plot(data2['NPP'], y_pred, color='blue')
    test_pred_y = np.array(data2['ModisNPP_x']).reshape(-1,1)
    regressor.fit(test_y, test_pred_y)
    y_pred = regressor.predict(test_y)
    plt.plot(data2['NPP'], y_pred, color='orange')
    ax.plot(data2['NPP'], data2['NPP'], linestyle='dotted', color='black')
    ax.set_xlabel("Training Data NPP (g C m-2) = ANPP + BNPP", fontweight='bold', fontsize=8)
    ax.set_ylabel("NPP CONUS (g C m-2)", fontweight='bold', fontsize=8)
    for label in ax.get_xticklabels():
        label.set_fontsize(8)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontsize(8)
        label.set_fontweight('bold')
    ax.set_title("Training Data vs Satellite NPP", fontweight='bold', fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig("files/NPP_Comparison.png", dpi=300)
    plt.close(fig)

df = pd.DataFrame({'Training_Data': list(data2['NPP']), 'Modis': list(data2['ModisNPP_x']), 'Landsat': list(data2['LandsatNPP_x'])})
df.to_csv("files/NPP_comparison.csv", index=False)

df = pd.read_csv("files/GHG_var_importance.csv")
feat_imp = df['Importance']
sorted_idx = np.argsort(feat_imp)
pos = np.arange(sorted_idx.shape[0]) + 0.5
with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(pos, feat_imp, align="center")
    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(df['Variable']), fontsize=14, fontweight='bold')
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
    ax.set_title("Heterotrophic Respiration", fontsize=16, fontweight='bold')
    # ax.set_title("Belowground Biomass", fontsize=16, fontweight='bold')
    ax.set_xlabel("Importance", fontsize=18, fontweight='bold')
    fig.tight_layout()
    fig.savefig("files/GHG_importance.png", dpi=300)
    plt.close(fig)

with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(6, 6))
    regressor = LinearRegression()
    test_y = np.array(y_test).reshape(-1,1)
    test_pred_y = np.array(y_test_pred).reshape(-1,1)
    regressor.fit(test_y, test_pred_y)
    y_pred = regressor.predict(test_y)
    ax.scatter(test_y, test_pred_y, color='g')
    ax.plot(test_y, y_pred, color='k', label='Regression line')
    ax.plot(test_y, test_y, linestyle='dotted', color='gray', label='1:1 line')
    ax.set_xlabel('Actual ' + y_field)
    ax.set_ylabel("Predicted " + y_field)
    ax.set_title("GHG Plot for Test data")
    # Make axes of equal extents
    axes_lim = np.ceil(max(max(test_y), max(test_pred_y))) + 2
    ax.set_xlim((0, axes_lim))
    ax.set_ylim((0, axes_lim))
    for label in ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontsize(10)
        label.set_fontweight('bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig("files/GHG_scatterplot.png", dpi=300)
    plt.close(fig)

df = pd.DataFrame({'Predicted': list(y_test_pred), 'Actual': y_test})
df.to_csv("files/GHG_scatter.csv", index=False)
df = pd.DataFrame({'Variable': list(np.array(ghg_model.feature_names_in_)[sorted_idx]), 'Importance': np.array(ghg_model.feature_importances_)[sorted_idx]})
df.to_csv("files/GHG_var_importance.csv", index=False)

NEP_mean = []
time_ser = range(1984, 2025)
for year in time_ser:
    data = pd.read_csv(f"files/{year}_Meadows_stats.csv")
    NEP_mean.append(data.NEP_mean.mean())
df = pd.DataFrame({'Year': list(time_ser), 'NEP': NEP_mean})
summary = df.groupby('Year')['NEP'].first()
with PdfPages('files/Histogram.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary.index.astype(str), summary.values, alpha=0.6)
    ax.plot(summary.index.astype(str), summary.values, marker='o', linewidth=2, label="Mean NEP")
    ax.set_xlabel("Year", fontweight='bold')
    ax.set_ylabel("NEP (g C m-2)", fontweight='bold')
    ax.set_title("NEP (Net Ecosystem Productivity) per Year", fontweight='bold')
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()