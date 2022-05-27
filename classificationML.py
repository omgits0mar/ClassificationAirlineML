# %%

# import jupyterthemes as jt
# from jupyterthemes import get_themes
# import jupyterthemes as jt
# from jupyterthemes.stylefx import set_nb_theme

# %% md

import jupyterthemes as jt
from jupyterthemes.stylefx import set_nb_theme

# %%

# set_nb_theme('onedork')
# #monokai
# #chesterish
# #oceans16 gamed
# #onedork gamed brdo
# #solarizedl

# %% md

# Code start from here :

# %% md

### Importing :

# %%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import datetime

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import joblib

# %% md

### Reading data from csv

# %%

df = pd.read_csv('airline-price-classification.csv')

# %%

df.head()

# %%

df.shape

# %% md

### Data preprocessing on 'price'

# %%

price_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=4)
df['TicketCategory'] = price_enc.fit_transform(df[["TicketCategory"]])
print(df['TicketCategory'])

# %%

filename = "price_enc.save"
joblib.dump(price_enc, filename)

# %% md

### Data preprocessing on 'date'

# %%

df["date"] = pd.to_datetime(df["date"])
df["date"] = df["date"].dt.strftime("%m/%d/%Y")

# %%

print(df[df.columns[0]])

# %%

pd.DatetimeIndex(df["date"]).weekday

# %%

df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
df['dayofyear'] = pd.DatetimeIndex(df['date']).dayofyear

# %%

# df = pd.get_dummies(df, columns=['month'], drop_first=True, prefix='month')

# %% md

### Data preprocessing : 'ch_code'

# %%

ch_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=9)
df['ch_code'] = ch_enc.fit_transform(df[["ch_code"]])
print(df['ch_code'])

# %%

filename = "ch_enc.save"
joblib.dump(ch_enc, filename)

# %% md

### Data preprocessing : 'dep_time'

# %%

df["dep_time"] = pd.to_datetime(df["dep_time"])
df['dep_time'] = df['dep_time'].dt.strftime("%-H:%M")

# %%

df["dep_hour"] = pd.DatetimeIndex(df["dep_time"]).hour
df["dep_minute"] = pd.DatetimeIndex(df["dep_time"]).minute

# %% md

### Data preprocessing : 'time_taken'

# %%

df["hours_taken"] = df["time_taken"].str.split('h').str.get(0)
df["minutes_taken"] = df["time_taken"].str[4:6]
df["minutes_taken"] = df["minutes_taken"].str.replace('m', '')
df["minutes_taken"] = df["minutes_taken"].str.replace('h', '')
df["hours_taken"] = pd.to_numeric(df["hours_taken"])
df["minutes_taken"] = pd.to_numeric(df["minutes_taken"], errors='coerce')
df.head()

# %%


# %% md

### Data preprocessing : 'stop'

# %%

df["stop"] = df["stop"].str.split('-').str.get(0)
df["stop"] = df["stop"].replace(['non'], 0)
df.isna().sum()  # 28944 null vals
df["stop"] = df["stop"].replace(['2+'], 2)  # Indicates for 2 or more stops
df['stop'] = df['stop'].fillna(0)
df['stop'] = pd.to_numeric(df['stop'])
# print(df[9:14])

# %% md

### Data preprocessing : 'arr_time'

# %%

df["arr_time"] = pd.to_datetime(df["arr_time"])
df['arr_time'] = df['arr_time'].dt.strftime("%-H:%M")
df["arr_hour"] = pd.DatetimeIndex(df["arr_time"]).hour
df["arr_minute"] = pd.DatetimeIndex(df["arr_time"]).minute
df["arr_hour"] = pd.to_numeric(df["arr_hour"])
df["arr_minute"] = pd.to_numeric(df["arr_minute"])
df.head()

# %% md

### Data preprocessing : 'type'

# %%

type_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=2)
df['type'] = type_enc.fit_transform(df[["type"]])
print(df['type'])

# %%

filename = "type_enc.save"
joblib.dump(type_enc, filename)

# %% md

### Data preprocessing : 'route'

# %%

df['source'] = df['route'].str.split(', ').str.get(0).str.split(':').str.get(1)
df['destination'] = df['route'].str.split(', ').str.get(1).str.split(':').str.get(1).str.split('}').str.get(0)
df['source'] = df['source'].str.replace('\'', "")
df['destination'] = df['destination'].str.replace('\'', "")

# %%

df.head()

# %%

source_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=6)
df['source'] = source_enc.fit_transform(df[["source"]])
print(df['source'])

# %%

filename = "source_enc.save"
joblib.dump(source_enc, filename)

# %%

destination_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=6)
df['destination'] = destination_enc.fit_transform(df[["destination"]])
print(df['destination'])
df = df.fillna(-1)
df = df.drop(['airline', 'date', 'dep_time', "time_taken", 'arr_time', 'route', ], axis=1)
# df = pd.get_dummies(df)

# %%

filename = "destination_enc.save"
joblib.dump(destination_enc, filename)

# %%

df.head()

# %%

print(df.columns)

# %%

X = df.loc[:, df.columns != 'TicketCategory']
Y = df['TicketCategory']

# %%


# %%

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=8)

# %%

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaler = ss.fit(XTrain)
trainX_scaled = scaler.transform(XTrain)
testX_scaled = scaler.transform(XTest)

# %%

scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)

# %%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

ridge = RidgeClassifier(alpha=0.1)
ridge.fit(trainX_scaled, YTrain)

rfc = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100)
rfc.fit(trainX_scaled, YTrain)

mlp_clf = MLPClassifier(hidden_layer_sizes=(150, 100, 50),
                        max_iter=50, activation='tanh',
                        solver='adam')

mlp_clf.fit(trainX_scaled, YTrain)

dtree = DecisionTreeClassifier(max_depth=50, random_state=42)
dtree.fit(trainX_scaled, YTrain)

knn = KNeighborsClassifier()
knn.fit(trainX_scaled, YTrain)

logestic = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs')
logestic.fit(trainX_scaled, YTrain)

# rbf_svc = svm.SVC(kernel='rbf', gamma='scale', C=1).fit(trainX_scaled, YTrain)
# poly_svc = svm.SVC(kernel='poly',degree = 3, C=1).fit(trainX_scaled, YTrain)

# %%

for i, clf in enumerate((ridge, rfc, mlp_clf, dtree, knn, logestic)):
    predictions = clf.predict(testX_scaled)
    print("Accuracy of: " + str(clf) + ":" + str(accuracy_score(YTest, predictions)))
    print('\n')
    print(classification_report(YTest, predictions))

# %%

from sklearn.metrics import plot_confusion_matrix

for i, clf in enumerate((ridge, rfc, mlp_clf, dtree, knn, logestic)):
    fig = plot_confusion_matrix(clf, testX_scaled, YTest, display_labels=clf.classes_)
    fig.figure_.suptitle("Confusion Matrix for: " + str(clf))
    plt.show()

# %% md


# %%

import joblib

filename = '_ClassificationModel.save'
for i, clf in enumerate((ridge, rfc, mlp_clf, dtree, knn, logestic)):
    joblib.dump(clf, str(clf) + filename)

# %%

# # some time later...

# # load the model from disk
# loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test, Y_test)
# print(result)

# %% md

## Scaling Applied :

### Accuracy of: MLPClassifier():0.8862869988759835
### Accuracy of: MLPClassifier(activation='relu', hidden_layer_sizes=(150, 100, 50), max_iter=50, solver='lbfgs'):0.7662670163606844
### Accuracy of: MLPClassifier(activation='tanh', hidden_layer_sizes=(150, 100, 50), max_iter=50,solver='sgd'):0.8349777278214895
### Accuracy of: MLPClassifier(activation='relu', hidden_layer_sizes=(150, 100, 50), max_iter=50,solver='adam'):0.9352233462387078
### Accuracy of: MLPClassifier(activation='tanh', hidden_layer_sizes=(150, 100, 50), max_iter=50,solver='adam'):0.9402397901835894
### Accuracy of: MLPClassifier(activation='tanh', hidden_layer_sizes=(150, 100, 50),max_iter=100,solver='adam'):0.9444444444444444 (take > 10 mins)

### Accuracy of: LogisticRegression(multi_class='multinomial', max_iter=100)  :0.7278423046500978
### Accuracy of: LogisticRegression(multi_class='multinomial', max_iter=50)   :0.715186711627326
### Accuracy of: LogisticRegression(C=0.01, solver='liblinear'):               0.689480038299821
### Accuracy of: LogisticRegression(C=0.1, solver='lbfgs'):0.7187044669247742

### Accuracy of: RidgeClassifier(alpha=0.1):0.7104824944839931


### Accuracy of: KNeighborsClassifier():0.8935098455518088

### Accuracy of: DecisionTreeClassifier(random_state=42):                       0.959056658756921
### Accuracy of: DecisionTreeClassifier(max_depth=20, random_state=42):        0.9547895591357562
### Accuracy of: DecisionTreeClassifier(max_depth=50, random_state=42):        0.959056658756921

### Accuracy of: RandomForestClassifier(n_jobs=-1, random_state=42):           0.9621581116523042-->BEST


# Conclusion :

# BIG NOTES :
### We tried performing both label, onehot encoder classifying them to 4 binary categories of the Y data, And was found that both results in fitting models gave nearly the same accuracy but label encoder had slight advantage due to inability of logestic reg. to fit on a 4D array of y labels in oneHot.

### all this predictions are gathered on GoogleCollab and it may slightly differs in jupyter or pycharm


# %% md


