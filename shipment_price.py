import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.filterwarnings('ignore')

# combine those two csv files into one main file 
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combined_df = pd.concat([train_df,test_df],axis = 0)
combined_df.to_csv('shipment_price.csv',index=False)
df = pd.read_csv('shipment_price.csv')

df = df.rename(columns={'Artist Reputation' :'Artist_Reputation','Customer Information':'Customer_Information'})



# Data Preprocessing
# # print(df['Customer Id'].unique())
df['Customer Id'] = df['Customer Id'].str.replace('f',"").str.replace("e",'')
df['Customer Id'] = df['Customer Id'].astype(float)

# # print(df['Scheduled Date'].unique())
df['Scheduled Date'] = pd.to_datetime(df['Scheduled Date'],errors='coerce')
df['Scheduled_year'] = df['Scheduled Date'].dt.year
df['Scheduled_month'] = df['Scheduled Date'].dt.month
df['Scheduled Day'] = df['Scheduled Date'].dt.day
df = df.drop('Scheduled Date',axis =1)

# # print(df['Delivery Date'].unique())
df['Delivery Date'] = pd.to_datetime(df['Delivery Date'],errors='coerce')
df['Delivery_year'] = df['Delivery Date'].dt.year
df['Delivery_month'] = df['Delivery Date'].dt.month
df['Delivery_day'] = df['Delivery Date'].dt.day
df = df.drop('Delivery Date',axis = 1)

# # data encoding
from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()

df['Artist Name']  = encoder.fit_transform(df['Artist Name'])
df['Artist Name'] = df['Artist Name'].astype(int)

df['Material']  = encoder.fit_transform(df['Material'])
df['Material'] = df['Material'].astype(int)

df['International']  = encoder.fit_transform(df['International'])
df['International'] = df['International'].astype(int)

df['Express Shipment']  = encoder.fit_transform(df['Express Shipment'])
df['Express Shipment'] = df['Express Shipment'].astype(int)

df['Installation Included']  = encoder.fit_transform(df['Installation Included'])
df['Installation Included'] = df['Installation Included'].astype(int)

df['Transport']  = encoder.fit_transform(df['Transport'])
df['Transport'] = df['Transport'].astype(int)

df['Fragile']  = encoder.fit_transform(df['Fragile'])
df['Fragile'] = df['Fragile'].astype(int)

df['Customer_Information']  = encoder.fit_transform(df['Customer_Information'])
df['Customer_Information'] = df['Customer_Information'].astype(int)

df['Remote Location']  = encoder.fit_transform(df['Remote Location'])
df['Remote Location'] = df['Remote Location'].astype(int)

df['Customer Location']  = encoder.fit_transform(df['Customer Location'])
df['Customer Location'] = df['Customer Location'].astype(int)

df['Cost'] = df['Cost'].fillna(df['Cost'].median())

nums = ['Customer Id', 'Artist Name', 'Artist_Reputation', 'Height', 'Width', 'Weight', 'Material', 'Price Of Sculpture', 'Base Shipping Price', 
        'International', 'Express Shipment', 'Installation Included', 'Transport', 'Fragile', 'Customer_Information', 'Remote Location', 
        'Customer Location', 'Scheduled_year', 'Scheduled_month', 'Scheduled Day', 'Delivery_year', 'Delivery_month', 'Delivery_day']

for i in nums:
    df[i] = df[i].fillna(df[i].median())

# feature selection
x = df.drop('Cost',axis = 1)
y = df['Cost']

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
rfe = RFE(estimator = LinearRegression(),n_features_to_select=5)
rfe.fit(x,y)
rfe.predict(x)

# # # print("original feature names",df.columns)
# # # print("rfe features names",rfe.support_)

df  = df.drop(columns= ['Customer Id', 'Artist Name','Height','Weight','Price Of Sculpture','Base Shipping Price','Express Shipment',
                        'Installation Included', 'Remote Location','Customer Location','Scheduled Day','Delivery_day'])



# #  Detect Outliers
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso_forest.fit_predict(df)
outliers = df[df['anomaly'] == -1]
# # print(f"Number of outliers detected: {len(outliers)}")
for col in df.columns:
    if df[col].dtype != 'object':
        median_val = df[col].median()
        df.loc[df['anomaly'] == -1,col] = median_val

df = df.drop('anomaly',axis  =1)

x = df.drop('Cost',axis = 1)
y = df['Cost']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# # # print(x_train.shape)
# # # print(y_train.shape)
# # # print(x_test.shape)
# # # print(y_test.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
model = RandomForestRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
# # print(r2_score(y_test,y_pred))

import pickle
pickle.dump(model,open('ship.pkl',"wb"))

