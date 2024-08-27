import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
import pickle

# # Loading the dataset

flood_train = pd.read_csv('train.csv')

# flood_train.head(10)
flood_train.shape
flood_train.info()
missing = {}
for i in range(0 , len(flood_train.isnull().sum().index)):
    if flood_train.isnull().sum()[i] != 0:
        missing[flood_train.isnull().sum().index[i]] = flood_train.isnull().sum()[i]
missing_data = pd.Series(missing, dtype=object).to_frame()
missing_data = missing_data.rename(columns={0:"missing values"})
        
missing_data  
flood_train.drop('id' , axis=1 , inplace=True)
def handle_outliers(df, threshold):
    for i in df.columns:
        z_score=zscore(df[i])
        outliers=df[i].loc[np.abs(z_score)>threshold]
        df.drop(outliers.index,inplace=True)
    return df
handle_outliers(flood_train, 2.75)
flood_train.drop_duplicates(inplace=True)

X = flood_train.drop(['FloodProbability'], axis=1)
y = flood_train['FloodProbability']

scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.fit_transform(X)
X_scaled = pd.DataFrame(scaled_data, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=52)

model=keras.Sequential()
model.add(layers.Dense(256, activation='relu',input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, validation_split=0.2, verbose=1)

pickle.dump(model ,open('model.pkl','wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))
# model=pickle.load(open('model.pkl','rb'))