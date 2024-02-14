import os, random
import pandas as pd
os.chdir("Code")

df = pd.read_csv("All_meadows_2022.csv")
r1, r2 = max(df['B5_mean']), min(df['B5_mean'])
step = (r1-r2)/3
low_NIR = df[df['B5_mean'] < r2+step]
mid_NIR = df[(df['B5_mean'] >= r2+step) & (df['B5_mean'] <= r2+2*step)]
high_NIR = df[df['B5_mean'] > r2+2*step]

random.seed(10)

l1, l2 = max(low_NIR['latitude']), min(low_NIR['latitude'])
step = (l1-l2)/3
low_lat_low_NIR = low_NIR[low_NIR['latitude'] < l2+step].sample(n=10)
mid_lat_low_NIR = low_NIR[(low_NIR['latitude'] >= l2+step) & 
                          (low_NIR['latitude'] <= l2+2*step)].sample(n=10)
high_lat_low_NIR = low_NIR[low_NIR['latitude'] > l2+2*step].sample(n=10)

l1, l2 = max(mid_NIR['latitude']), min(mid_NIR['latitude'])
step = (l1-l2)/3
low_lat_mid_NIR = mid_NIR[mid_NIR['latitude'] < l2+step].sample(n=10)
mid_lat_mid_NIR = mid_NIR[(mid_NIR['latitude'] >= l2+step) & 
                          (mid_NIR['latitude'] <= l2+2*step)].sample(n=10)
high_lat_mid_NIR = mid_NIR[mid_NIR['latitude'] > l2+2*step].sample(n=10)

l1, l2 = max(high_NIR['latitude']), min(high_NIR['latitude'])
step = (l1-l2)/3
low_lat_high_NIR = high_NIR[high_NIR['latitude'] < l2+step].sample(n=10)
mid_lat_high_NIR = high_NIR[(high_NIR['latitude'] >= l2+step) & 
                            (high_NIR['latitude'] <= l2+2*step)].sample(n=10)
high_lat_high_NIR = high_NIR[high_NIR['latitude'] > l2+2*step].sample(n=10)

frames = [low_lat_low_NIR, mid_lat_low_NIR, high_lat_low_NIR, low_lat_mid_NIR, mid_lat_mid_NIR, 
          high_lat_mid_NIR, low_lat_high_NIR, mid_lat_high_NIR, high_lat_high_NIR]

data = pd.concat(frames)
data.to_csv('training_and_test_data.csv', index=False)


from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform

data = pd.read_csv("training_and_test_data.csv")
var_col = [c for c in data if c not in ['ID', 'QA_PIXEL_mean', 'QA_PIXEL_variance', 'Is_meadow']]
X = data.loc[:, var_col]
Y = data.loc[:, 'Is_meadow']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
Y.value_counts()

parameters = {'learning_rate': uniform(), 'subsample': uniform(), 
              'n_estimators': randint(50, 5000), 'max_depth': randint(2, 10)}
randm = RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=parameters,
                           cv=10, n_iter=20, n_jobs=1)
randm.fit(X_train, y_train)
randm.best_estimator_
randm.best_params_

parameters = {'learning_rate': [0.01, 0.03, 0.05], 'subsample': [0.4, 0.5, 0.6], 
              'n_estimators': [1000, 2500, 5000], 'max_depth': [6,7,8]}
grid = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=parameters, cv=10, n_jobs=1)
grid.fit(X_train, y_train)
grid.best_params_

gbm_model = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, n_estimators=2500, subsample=0.6,
                                       validation_fraction=0.1, n_iter_no_change=20, max_features='log2',
                                       verbose=1, random_state=48)
gbm_model.fit(X_train, y_train)
gbm_model.score(X_test, y_test)
len(gbm_model.estimators_)

y_train_pred = gbm_model.predict_proba(X_train)[:, 1]
y_test_pred = gbm_model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_train, y_train_pred))
print(roc_auc_score(y_test, y_test_pred))
