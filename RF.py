import numpy as np
import pandas as pd
#from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
import gc

print('Loading data ...')

train = pd.read_csv('./input/train_2016.csv')
prop = pd.read_csv('./input/properties_2016.csv')
sample = pd.read_csv('./input/sample_submission.csv')

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]


imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
x_train = imp.fit_transform(x_train)
x_valid = imp.transform(x_valid)
print type(x_train)

print('Training RF...')

#clf = svm.SVR()
clf = RandomForestRegressor(n_estimators=100, criterion='mae',max_depth=4,max_features='sqrt',verbose=2,n_jobs=-1)
clf.fit(x_train, y_train) 
y_pred = clf.predict(x_valid)

print('MAE = {0:.7f}'.format(mean_absolute_error(y_valid, y_pred)))

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

del prop; gc.collect()

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test, sample; gc.collect()

print('Predicting on test ...')
x_test = imp.transform(x_test)
y_test = clf.predict(x_test)

sub = pd.read_csv('./input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = y_test

print('Writing csv ...')
sub.to_csv('rf_yo.csv', index=False, float_format='%.4f') # Thanks to @inversion
print('Done')