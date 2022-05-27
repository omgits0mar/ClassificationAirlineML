#%%
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
from sklearn.preprocessing import StandardScaler
import joblib

#%%
df = pd.read_csv('Samples/airline-test-samples.csv')
df.head()
#%%
import warnings
warnings.filterwarnings("ignore")

#%% md
#### Loading TicketCategory Encoder :
#%%
price_enc = joblib.load("EncoderModels/price_enc.save")
df['TicketCategory'] = price_enc.transform(df[["TicketCategory"]])
df.head()

#%% md
#### Data preprocessing on Date
#%%
df["date"]=pd.to_datetime(df["date"])
df["date"]=df["date"].dt.strftime("%m/%d/%Y")
pd.DatetimeIndex(df["date"]).weekday
df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
df['dayofyear'] = pd.DatetimeIndex(df['date']).dayofyear

#%% md
#### Loading Ch_code Encoder :
#%%
ch_enc = joblib.load("EncoderModels/ch_enc.save")

df['ch_code'] = ch_enc.transform(df[["ch_code"]])

df.head()
#%% md
#### Departure time preprocessing
#%%
df["dep_time"]=pd.to_datetime(df["dep_time"])
df['dep_time'] = df['dep_time'].dt.strftime("%-H:%M")
df["dep_hour"]=pd.DatetimeIndex(df["dep_time"]).hour
df["dep_minute"]=pd.DatetimeIndex(df["dep_time"]).minute
#%% md
#### Time_taken preprocessing
#%%
df["hours_taken"] = df["time_taken"].str.split('h').str.get(0)
df["minutes_taken"] = df["time_taken"].str[4:6]
df["minutes_taken"] = df["minutes_taken"].str.replace('m', '')
df["minutes_taken"] = df["minutes_taken"].str.replace('h', '')
df["hours_taken"] = pd.to_numeric(df["hours_taken"])
df["minutes_taken"] = pd.to_numeric(df["minutes_taken"], errors='coerce')
#%% md
#### Stop preprocessing
#%%
df["stop"] = df["stop"].str.split('-').str.get(0)
df["stop"] = df["stop"].replace(['non'], 0)
df.isna().sum() #  28944 null vals
df["stop"] = df["stop"].replace(['2+'], 2) # Indicates for 2 or more stops
df['stop'] = df['stop'].fillna(0)
df['stop'] = pd.to_numeric(df['stop'])
#%% md
#### Arrival time preprocessing
#%%
df["arr_time"]=pd.to_datetime(df["arr_time"])
df['arr_time'] = df['arr_time'].dt.strftime("%-H:%M")
df["arr_hour"]=pd.DatetimeIndex(df["arr_time"]).hour
df["arr_minute"]=pd.DatetimeIndex(df["arr_time"]).minute
df["arr_hour"] = pd.to_numeric(df["arr_hour"])
df["arr_minute"] = pd.to_numeric(df["arr_minute"])
#%% md
#### Source & Destination preprocessing
#%%
df['source'] = df['route'].str.split( ', ').str.get(0).str.split(':').str.get(1)
df['destination'] = df['route'].str.split( ', ').str.get(1).str.split(':').str.get(1).str.split('}').str.get(0)
df['source'] = df['source'].str.replace('\'', "")
df['destination'] = df['destination'].str.replace('\'', "")
#%% md
#### Loading Type Encoder :
#%%
type_enc = joblib.load("EncoderModels/type_enc.save")
df['type'] = type_enc.transform(df[["type"]])
df.head()
#%% md
#### Loading Source Encoder :
#%%
source_enc = joblib.load("EncoderModels/source_enc.save")
df['source'] = source_enc.transform(df[["source"]])
df.head()
#%% md
#### Loading Destination Encoder :
#%%
destination_enc = joblib.load("EncoderModels/destination_enc.save")
df['destination'] = source_enc.transform(df[["destination"]])
df.head()
#%% md
#### Cleaning Data
#%%
df = df.fillna(-1)
df = df.drop(['airline', 'date', 'dep_time', "time_taken", 'arr_time', 'route',], axis=1)
df.head()
#%%
X = df.loc[:, df.columns != 'TicketCategory']
Y = df['TicketCategory']
#%% md
#### Loading Scaler Model :
#%%
scaler = joblib.load("scaler.save")
X_scaled = scaler.transform(X)
#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
#%% md
#### Loading Classification model :
#%%
rfc_model = joblib.load("PredictionModels\RandomForestClassifier(n_jobs=-1, random_state=42)_ClassificationModel.save")
dtc_model = joblib.load("PredictionModels\DecisionTreeClassifier(max_depth=50, random_state=42)_ClassificationModel.save")
knn_model = joblib.load("PredictionModels\KNeighborsClassifier()_ClassificationModel.save")
logestic_model = joblib.load("PredictionModels\LogisticRegression(C=0.1)_ClassificationModel.save")
mlpc_model = joblib.load("PredictionModels\MLPClassifier(activation='tanh', hidden_layer_sizes=(150, 100, 50), max_iter=50)_ClassificationModel.save")
ridge_model = joblib.load("PredictionModels\RidgeClassifier(alpha=0.1)_ClassificationModel.save")

#%%
for i, clf in enumerate((rfc_model, dtc_model, knn_model, logestic_model, mlpc_model, ridge_model)):
    predictions = clf.predict(X_scaled)
    print("Accuracy of: " + str(clf)+":"+str(accuracy_score(Y, predictions)))
    print('\n')
    print(classification_report(Y, predictions))
