# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:55:04 2024

@author: Johanson C. Onyegbula
"""

import itertools
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

# define "grid values" of hyperparameters for iterative optimization
parameters = {'subsample': [x/100 for x in range(30, 101, 10)], 'max_depth': range(3,15)}
learning_rate = [0.01,0.02,0.03,0.05,0.07,0.08,0.1,0.13,0.16,0.2,0.25,0.3,0.35]
n_estimators = [25,50,75,100,125,150,200,250,500,750,1000,2000,2500,5000]

def trainModel(alph, n_est):
    hypertune = {'Mean_RMSE': [float('Inf')], 'Test_RMSE': [float('Inf')]}
    count = 0
    
    for sub in parameters['subsample']:
        for depth in parameters['max_depth']:
            # use hyperparameter combination to fit model (or alternative 1 SD test model below)
            gbm_model = GradientBoostingRegressor(learning_rate=alph, max_depth=depth, n_estimators=n_est, subsample=sub,
                                                   validation_fraction=0.2, n_iter_no_change=50, max_features='log2',
                                                   random_state=10)
            '''
            gbm_model = GradientBoostingRegressor(loss="quantile", learning_rate=alph, alpha=0.8413, max_depth=depth, 
                                                  n_estimators=n_est, subsample=sub, validation_fraction=0.2, n_iter_no_change=50,  
                                                  max_features='log2', random_state=10)'''
            gbm_model.fit(X_train, y_train)
            
            # compute relevant accuracy stats of actual model and test performance of hyperparameter combination
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

# use less threads than available in parallel computing to keep computer from freezing
with Parallel(n_jobs=18, prefer="threads") as parallel:
    result = parallel(delayed(trainModel)(alph, n_est) for alph, n_est in list(itertools.product(learning_rate, n_estimators)))

# print out final accuracy stats with optimized hyperparameters
hypertune = {'Mean_RMSE': [float('Inf')], 'Test_RMSE': [float('Inf')]}
for x in result:
    if x['Mean_RMSE'][0] < hypertune['Mean_RMSE'][0]:
        hypertune['Mean_RMSE'] = x['Mean_RMSE']
    if x['Test_RMSE'][0] < hypertune['Test_RMSE'][0]:
        hypertune['Test_RMSE'] = x['Test_RMSE']
hypertune


# train classification model
def trainClassifier(alph, n_est):
    hypertune = {'Mean_Score': [0], 'Test_Score': [0]}
    count = 0
    
    for sub in parameters['subsample']:
        for depth in parameters['max_depth']:
            # use hyperparameter combination to fit model (or alternative 1 SD test model below)
            gbm_model = GradientBoostingClassifier(learning_rate=alph, max_depth=depth, n_estimators=n_est, subsample=sub,
                                                   validation_fraction=0.2, n_iter_no_change=50, max_features='log2',
                                                   random_state=10)
            gbm_model.fit(X_train, y_train)
            
            # compute relevant accuracy stats of actual model and test performance of hyperparameter combination
            y_train_pred = gbm_model.predict_proba(X_train)[:, 1]
            y_test_pred = gbm_model.predict_proba(X_test)[:, 1]
            train_score = roc_auc_score(y_train, y_train_pred)
            test_score = roc_auc_score(y_test, y_test_pred)
            mean_score = (train_score + test_score)/2
            
            if mean_score > hypertune['Mean_Score'][0]:
                hypertune['Mean_Score'] = [mean_score, alph, n_est, sub, depth, 0.2]
            if test_score > hypertune['Test_Score'][0]:
                hypertune['Test_Score'] = [test_score, alph, n_est, sub, depth, 0.2]
           
            count += 1
            if count%40 == 0:
                print(count, end=' ')
    return hypertune

# use less threads than available in parallel computing to keep computer from freezing
with Parallel(n_jobs=18, prefer="threads") as parallel:
    result = parallel(delayed(trainClassifier)(alph, n_est) for alph, n_est in list(itertools.product(learning_rate, n_estimators)))

# print out final accuracy stats with optimized hyperparameters
hypertune = {'Mean_Score': [0], 'Test_Score': [0]}
for x in result:
    if x['Mean_Score'][0] > hypertune['Mean_Score'][0]:
        hypertune['Mean_Score'] = x['Mean_Score']
    if x['Test_Score'][0] > hypertune['Test_Score'][0]:
        hypertune['Test_Score'] = x['Test_Score']
hypertune
