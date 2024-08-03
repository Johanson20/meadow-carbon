# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:55:04 2024

@author: jonyegbula
"""

import statistics
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

parameters = {'learning_rate': [0.01,0.03,0.07,0.1,0.13,0.16,0.2,0.25,0.3,0.35], 'subsample': [x/100 for x in range(30, 101, 10)],
             'n_estimators': [25,50,75,100,125,150,200,250,500,750,1000], 'max_depth': range(3,15)}

hypertune = {'RMSE': [float('Inf')], 'BIAS': [float('Inf')]}
count = 0

for sub in parameters['subsample']:
    for alpha in parameters['learning_rate']:
        for n_est in parameters['n_estimators']:
            for depth in parameters['max_depth']:
                gbm_model = GradientBoostingRegressor(learning_rate=alpha, max_depth=depth, n_estimators=n_est, subsample=sub,
                                                       validation_fraction=0.2, n_iter_no_change=50, max_features='log2',
                                                       random_state=48)
                gbm_model.fit(X_train, y_train)
                y_train_pred = gbm_model.predict(X_train)
                y_test_pred = gbm_model.predict(X_test)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                val = (y_train_pred - y_train) / y_train
                train_p_bias = abs(np.mean(val[np.isfinite(val)]) * 100)
                
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                val = (y_test_pred - y_test) / y_test
                test_p_bias = abs(np.mean(val[np.isfinite(val)]) * 100)
                
                mean_rmse = round(statistics.harmonic_mean([train_rmse, test_rmse]), 5)
                mean_bias = round(statistics.harmonic_mean([train_p_bias, test_p_bias]), 5)
                
                if mean_rmse < hypertune['RMSE'][0]:
                    hypertune['RMSE'] = [mean_rmse, alpha, n_est, sub, depth, 0.2]
                if mean_bias < hypertune['BIAS'][0]:
                    hypertune['BIAS'] = [mean_bias, alpha, n_est, sub, depth, 0.2]
                
                count += 1
                if count%100 == 0:
                    print(count, end=' ')