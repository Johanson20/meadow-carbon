# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:08:10 2025

@author: jonyegbula
"""

from xgboostlss.model import *
from xgboostlss.distributions.Gaussian import *
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np

data = pd.read_csv("csv/Belowground Biomass_RS Model_Data.csv")
data.head()
# confirm column names first
cols = data.columns
# cols = data.columns[1:]     # drops unnecessary 'Unnamed: 0' column
data = data.loc[:, cols]
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)

# remove irrelevant columns for ML and determine X and Y variables
var_col =  list(cols[20:26]) + list(cols[-29:])
y_field = 'Roots.kg.m2'
# subdata excludes other measured values which can be largely missing (as we need to assess just one output at a time)
subdata = data.loc[:, ([y_field] + var_col)]

# check for missing/null values across columns and rows respectively (confirm results below should be all 0)
sum(subdata.isnull().any(axis=0) == True)
sum(subdata[y_field].isnull())

nullIds = list(np.where(subdata[y_field].isnull())[0])    # null IDs
data.drop(nullIds, inplace = True)
data.dropna(subset=[y_field], inplace=True)
data.dropna(subset=var_col, inplace=True)
data.reset_index(drop=True, inplace=True)

gsp = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=10)
split = gsp.split(data, groups=data['ID'])
train_index, test_index = next(split)
train_data = data.iloc[train_index]
test_data = data.iloc[test_index]

X_train, y_train = train_data.loc[:, var_col], train_data[y_field]
X_test, y_test = test_data.loc[:, var_col], test_data[y_field]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
xgblss = XGBoostLSS(Gamma(stabilization="L2", response_fn="exp", loss_fn="nll"))

param_dict = {
    "eta":              ["float", {"low": 1e-5,   "high": 1,     "log": True}],
    "max_depth":        ["int",   {"low": 1,      "high": 10,    "log": False}],
    "gamma":            ["float", {"low": 1e-8,   "high": 40,    "log": True}],
    "subsample":        ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
    "colsample_bytree": ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
    "min_child_weight": ["float", {"low": 1e-8,   "high": 500,   "log": True}],
    "booster":          ["categorical", ["gbtree"]]
}

np.random.seed(10)
opt_param = xgblss.hyper_opt(param_dict, dtrain, num_boost_round=100, nfold=5, early_stopping_rounds=20,
                             max_minutes=10, n_trials=None, silence=True, seed=10, hp_seed=None)
np.random.seed(10)
opt_params = opt_param.copy()
n_rounds = opt_params["opt_rounds"]
del opt_params["opt_rounds"]
xgblss.train(opt_params, dtrain, num_boost_round=n_rounds)

torch.manual_seed(10)
# Number of samples to draw from predicted distribution
n_samples = 100
quant_sel = [0.05, 0.16, 0.5, 0.84, 0.95] # Quantiles to calculate from predicted distribution
# Sample from predicted distribution
pred_samples = xgblss.predict(dtest, pred_type="samples", n_samples=n_samples, seed=10)

pred_params = xgblss.predict(dtest, pred_type="parameters")
y_test_pred = pred_params['loc']
y_train_pred = xgblss.predict(xgb.DMatrix(X_train), pred_type="parameters")['loc']
