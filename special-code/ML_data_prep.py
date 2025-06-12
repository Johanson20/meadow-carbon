# -*- coding: utf-8 -*-
"""
@author: Johanson C. Onyegbula
"""

import os, random
import pandas as pd
import geopandas as gpd
os.chdir("Code")    # change path to where github code is pulled from

# read in landsat/flow data for all 18,144 meadows
df = pd.read_csv("csv/Real_and_false_meadows.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# stratify meadows in 3 classes based on range of NIR values
r1, r2 = max(df['NIR_mean']), min(df['NIR_mean'])
step = (r1-r2)/3
low_NIR = df[df['NIR_mean'] < r2+step]
mid_NIR = df[(df['NIR_mean'] >= r2+step) & (df['NIR_mean'] <= r2+2*step)]
high_NIR = df[df['NIR_mean'] > r2+2*step]

# set random seed for reproducibility of random draws (a total of 800 random samples)
random.seed(10)

# stratify low NIR meadows into 3 latitude classes and draw 200 random samples each
l1, l2 = max(low_NIR['Latitude']), min(low_NIR['Latitude'])
step = (l1-l2)/3
low_lat_low_NIR = low_NIR[low_NIR['Latitude'] < l2+step].sample(n=200)
mid_lat_low_NIR = low_NIR[(low_NIR['Latitude'] >= l2+step) & (low_NIR['Latitude'] <= l2+2*step)].sample(n=200)
high_lat_low_NIR = low_NIR[low_NIR['Latitude'] > l2+2*step].sample(n=200)

# stratify mid NIR meadows into 3 latitude classes and draw 200 random samples each
l1, l2 = max(mid_NIR['Latitude']), min(mid_NIR['Latitude'])
step = (l1-l2)/3
low_lat_mid_NIR = mid_NIR[mid_NIR['Latitude'] < l2+step].sample(n=200)
mid_lat_mid_NIR = mid_NIR[(mid_NIR['Latitude'] >= l2+step) & (mid_NIR['Latitude'] <= l2+2*step)].sample(n=200)
high_lat_mid_NIR = mid_NIR[mid_NIR['Latitude'] > l2+2*step].sample(n=200)

# stratify high NIR meadows into 3 latitude classes and draw 200 random samples each
l1, l2 = max(high_NIR['Latitude']), min(high_NIR['Latitude'])
step = (l1-l2)/3
low_lat_high_NIR = high_NIR[high_NIR['Latitude'] < l2+step].sample(n=200)
mid_lat_high_NIR = high_NIR[(high_NIR['Latitude'] >= l2+step) & (high_NIR['Latitude'] <= l2+2*step)].sample(n=200)
high_lat_high_NIR = high_NIR[high_NIR['Latitude'] > l2+2*step].sample(n=200)

# combine all 1800 samples (200 for each sub-division of NIR and lat) into a single dataframe
frames = [low_lat_low_NIR, mid_lat_low_NIR, high_lat_low_NIR, low_lat_mid_NIR, mid_lat_mid_NIR, 
          high_lat_mid_NIR, low_lat_high_NIR, mid_lat_high_NIR, high_lat_high_NIR]
data = pd.concat(frames)
data.to_csv('csv/meadow_ID_training_data.csv', index=False)


# import relevant ML modules and classes
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import randint, uniform
import numpy as np

# read csv containing random samples
epsg_crs = "EPSG:4326"
meadows = gpd.read_file("files/AllPossibleMeadows_2025-06-06.shp").to_crs(epsg_crs)
data = pd.read_csv("csv/meadow_ID_training_data.csv")
cols = data.columns
# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in cols if c not in ['ID', 'Area_m2', 'Longitude', 'Latitude', 'IsMeadow']]
X = data.loc[:, var_col]
Y = data.loc[:, 'IsMeadow']

# split X and Y into training (80%) and test data (20%), random state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
Y.value_counts()

''' Before using gradient boosting, optimize hyperparameters either:
    by randomnly selecting from a range of values using RandomizedSearchCV,
    or by grid search which searches through all provided possibilities with GridSearchCV
'''
# optimize hyperparameters with RandomizedSearchCV on the training data (takes 30ish minutes)
parameters = {'learning_rate': uniform(), 'subsample': uniform(), 
              'n_estimators': randint(50, 5000), 'max_depth': randint(2, 10)}
randm = RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=parameters,
                           cv=10, n_iter=20, n_jobs=1)
randm.fit(X_train, y_train)
randm.best_estimator_
randm.best_params_      # outputs all parameters of ideal estimator

# same process above but with GridSearchCV for comparison (takes even longer)
parameters = {'learning_rate': [0.01, 0.03, 0.05], 'subsample': [0.4, 0.5, 0.6], 
              'n_estimators': [1000, 2500, 5000], 'max_depth': [6,7,8]}
grid = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=parameters, cv=10, n_jobs=1)
grid.fit(X_train, y_train)
grid.best_params_

# run gradient boosting with optimized parameters (chosen with hyperparameter_tuning) on training data
gbm_model = GradientBoostingClassifier(learning_rate=0.03, max_depth=3, n_estimators=200, subsample=0.4,
                                       validation_fraction=0.1, n_iter_no_change=20, max_features='log2',
                                       verbose=1, random_state=48)
gbm_model.fit(X_train, y_train)
gbm_model.score(X_test, y_test)  # mean accuracy (percentage/100)
len(gbm_model.estimators_)  # number of trees used in estimation
gbm_model.feature_importances_
gbm_model.feature_names_in_

# probability vectors on test data: "No" = [:, 0] and "Yes" = [:, 1], and roc_auc
y_train_pred = gbm_model.predict_proba(X_train)[:, 1]
y_test_pred = gbm_model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_train, y_train_pred))
print(roc_auc_score(y_test, y_test_pred))

# Compare training labels and compute probability vectors
data['MeadowProb'] = np.round(gbm_model.predict_proba(X)[:, 1], 5)
df['MeadowProb'] = np.round(gbm_model.predict_proba(df.loc[:, var_col])[:, 1], 5)
# extract probability vectors for real meadows (non-zero area in square meters)
realMeadows = meadows[meadows['ID'].isin(set(df[df['Area_m2'] > 0]['ID']))]
realMeadows['MeadowProb'] = np.round(gbm_model.predict_proba(df[df.Area_m2 > 0].loc[:, var_col])[:, 1], 5)
realMeadows.to_file("files/MeadowsProbability.shp", driver="ESRI Shapefile")
meadows = None

# assign 1 to "Yes" and "No" to 0 (most "Yes" have prob > 0.67 in training) on test data
y_pred = [1 if x=="Yes" else 0 for x in y_test]
predictions = [round(value-0.17) for value in y_test_pred]  # assuming 0.67 is meadow threshold

# compute accuracy and rmse manually (not reliable)
accuracy = sum([1 if y_pred[i]==predictions[i] else 0 for i in range(len(y_pred))])/len(y_pred)*100
rmse = np.sqrt(mean_squared_error(predictions, y_pred))

# compute confusion matrix related metrics
precision = precision_score(y_pred, predictions, pos_label=1)
recall = recall_score(y_pred, predictions, pos_label=1)
f1 = f1_score(y_pred, predictions, pos_label=1)
conf_matrix = confusion_matrix(y_pred, predictions, labels=[1, 0])

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)


# read in meadow data again without NaNs
df = pd.read_csv("csv/All_meadows_2022.csv").dropna()
# drop columns without landsat data and run predictions
X_vars = df.loc[:, var_col]
meadow_pred = gbm_model.predict_proba(X_vars)[:, 1]
# compute number of positively predicted meadows
noMeadows = sum([round(value-0.17) for value in meadow_pred])
X_vars.shape[0] - noMeadows     # number of False meadows


# subset meadows with unclear probability vectors (0.4 - 0.65 chosen)
obscure_data = df.loc[(meadow_pred > 0.4) & (meadow_pred < 0.65)]
obscure_data = obscure_data.assign(IsMeadow = None)

# repeat NIR stratification for sampling additional meadows
r1, r2 = max(obscure_data['NIR_mean']), min(obscure_data['NIR_mean'])
step = (r1-r2)/3
low_NIR = obscure_data[obscure_data['NIR_mean'] < r2+step]
mid_NIR = obscure_data[(obscure_data['NIR_mean'] >= r2+step) & (obscure_data['NIR_mean'] <= r2+2*step)]
high_NIR = obscure_data[obscure_data['NIR_mean'] > r2+2*step]

random.seed(48)

# stratify low NIR meadows into 3 latitude classes and draw 10 random samples each
l1, l2 = max(low_NIR['Latitude']), min(low_NIR['Latitude'])
step = (l1-l2)/3
low_lat_low_NIR = low_NIR[low_NIR['Latitude'] < l2+step].sample(n=10)
mid_lat_low_NIR = low_NIR[(low_NIR['Latitude'] >= l2+step) & (low_NIR['Latitude'] <= l2+2*step)].sample(n=10)
high_lat_low_NIR = low_NIR[low_NIR['Latitude'] > l2+2*step].sample(n=10)

# stratify mid NIR meadows into 3 latitude classes and draw 10 random samples each
l1, l2 = max(mid_NIR['Latitude']), min(mid_NIR['Latitude'])
step = (l1-l2)/3
low_lat_mid_NIR = mid_NIR[mid_NIR['Latitude'] < l2+step].sample(n=10)
mid_lat_mid_NIR = mid_NIR[(mid_NIR['Latitude'] >= l2+step) & (mid_NIR['Latitude'] <= l2+2*step)].sample(n=10)
high_lat_mid_NIR = mid_NIR[mid_NIR['Latitude'] > l2+2*step].sample(n=10)

# stratify high NIR meadows into 3 latitude classes and draw 10 random samples each
l1, l2 = max(high_NIR['Latitude']), min(high_NIR['Latitude'])
step = (l1-l2)/3
# each draws 10 unique rows not already in "data"
low_lat_high_NIR = high_NIR[high_NIR['Latitude'] < l2+step].sample(11)
mid_lat_high_NIR = high_NIR[(high_NIR['Latitude'] >= l2+step) & (high_NIR['Latitude'] <= l2+2*step)].sample (11)
high_lat_high_NIR = high_NIR[high_NIR['Latitude'] > l2+2*step].sample(11)

frames = [data, low_lat_low_NIR, mid_lat_low_NIR, high_lat_low_NIR, low_lat_mid_NIR, mid_lat_mid_NIR, 
          high_lat_mid_NIR, low_lat_high_NIR, mid_lat_high_NIR, high_lat_high_NIR]
newdf = pd.concat(frames).drop_duplicates(subset=['ID'])
newdf.to_csv('csv/new_training_and_test_data.csv', index=False)



# re-run ML for 180 samples
data = pd.read_csv("csv/new_training_and_test_data.csv")
data.head()
# remove irrelevant columns for ML and determine X and Y variables
var_col = [c for c in data if c not in ['ID', 'Longitude', 'Latitude', 'QA_PIXEL_mean', 'QA_PIXEL_variance', 'IsMeadow']]
X = data.loc[:, var_col] # var_col are all columns in csv except above 4
Y = data.loc[:, 'IsMeadow']

# split X and Y into training (80%) and test data (20%), random state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
Y.value_counts()

# Hyperparameter tuning with RandomizedSearchCV
parameters = {'learning_rate': [0.01, 0.03, 0.05], 'subsample': [0.4, 0.5, 0.6], 
              'n_estimators': [100, 200, 500, 1000, 2500, 5000], 'max_depth': [3,4,5,6]}
randm = RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=parameters,
                           cv=10, n_iter=20, n_jobs=1)
randm.fit(X_train, y_train)
randm.best_params_

# run gradient boosting with optimized parameters (chosen with GridSearchCV) on training data
gbm_model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.5,
                                       validation_fraction=0.1, n_iter_no_change=20, max_features='log2',
                                       verbose=1, random_state=48)
gbm_model.fit(X_train, y_train)
gbm_model.score(X_test, y_test)  # mean accuracy (percentage/100)
len(gbm_model.estimators_)  # number of trees used in estimation
gbm_model.feature_importances_
gbm_model.feature_names_in_

# probability vectors on test data: "No" = [:, 0] and "Yes" = [:, 1], and roc_auc
y_train_pred = gbm_model.predict_proba(X_train)[:, 1]
y_test_pred = gbm_model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_train, y_train_pred))
print(roc_auc_score(y_test, y_test_pred))

# Compare training labels and probability vectors
for i, x in enumerate(y_test):
    print(x, y_test_pred[i])

# read in meadow data again without NaNs
df = pd.read_csv("csv/All_meadows_2022.csv").dropna()
# drop columns without landsat data and run predictions
X_vars = df.loc[:, var_col]
meadow_pred = gbm_model.predict_proba(X_vars)[:, 1]
df['meadow_pred'] = meadow_pred
for threshold in [0.5, 0.6, 0.67]:
    name = 'csv/False_meadow_threshold_' + str(threshold) + ".csv"
    noMeadows = df.loc[df['meadow_pred'] < threshold]
    noMeadows.to_csv(name, index=False)
    print("Number of False meadows at threshold =", threshold, ":", noMeadows.shape[0])