# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:55:04 2024

@author: jonyegbula
"""

import itertools
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

parameters = {'subsample': [x/100 for x in range(30, 101, 10)], 'max_depth': range(3,15)}
learning_rate = [0.01,0.02,0.03,0.05,0.07,0.08,0.1,0.13,0.16,0.2,0.25,0.3,0.35]
n_estimators = [25,50,75,100,125,150,200,250,500,750,1000,2000,2500,5000]

def trainModel(alph, n_est):
    hypertune = {'Mean_RMSE': [float('Inf')], 'Test_RMSE': [float('Inf')]}
    count = 0
    
    for sub in parameters['subsample']:
        for depth in parameters['max_depth']:
            gbm_model = GradientBoostingRegressor(learning_rate=alph, max_depth=depth, n_estimators=n_est, subsample=sub,
                                                   validation_fraction=0.2, n_iter_no_change=50, max_features='log2',
                                                   random_state=10)
            '''
            gbm_model = GradientBoostingRegressor(loss="quantile", learning_rate=alph, alpha=0.8413, max_depth=depth, 
                                                  n_estimators=n_est, subsample=sub, validation_fraction=0.2, n_iter_no_change=50,  
                                                  max_features='log2', random_state=10)'''
            gbm_model.fit(X_train, y_train)
            
            y_train_pred = gbm_model.predict(X_train)
            y_test_pred = gbm_model.predict(X_test)
            train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
            mean_rmse = (train_rmse + test_rmse)/2
           
            if mean_rmse < hypertune['Mean_RMSE'][0]:
                hypertune['Mean_RMSE'] = [mean_rmse, alph, n_est, sub, depth, 0.2]
            if test_rmse < hypertune['Test_RMSE'][0]:
                hypertune['Test_RMSE'] = [test_rmse, alph, n_est, sub, depth, 0.2]
           
            count += 1
            if count%40 == 0:
                print(count, end=' ')
    return hypertune


with Parallel(n_jobs=18, prefer="threads") as parallel:
    result = parallel(delayed(trainModel)(alph, n_est) for alph, n_est in list(itertools.product(learning_rate, n_estimators)))
    
hypertune = {'Mean_RMSE': [float('Inf')], 'Test_RMSE': [float('Inf')]}
for x in result:
    if x['Mean_RMSE'][0] < hypertune['Mean_RMSE'][0]:
        hypertune['Mean_RMSE'] = x['Mean_RMSE']
    if x['Test_RMSE'][0] < hypertune['Test_RMSE'][0]:
        hypertune['Test_RMSE'] = x['Test_RMSE']
hypertune
