import numpy as np
import pandas as pd
import streamlit as st
import joblib

import warnings
warnings.filterwarnings('ignore')



@st.cache_data
def load_data():
    df = pd.read_csv('shipment_price.csv')
    return df

df = load_data()

ship = joblib.load('ship.pkl')




pages ={
    "Prediction":"Prediction Page",
    "Visualization": "Visualization Page"
}
page = st.sidebar.radio("Select Option:",options=list(pages.keys()))

st.sidebar.title("🚢 Industrial Applications of Shipment Price Detection")
st.sidebar.markdown(
    """
    - **📦 E-commerce & Logistics**: Optimizes shipping costs for online retailers by predicting the most cost-effective delivery options.
    - **🚛 Freight & Transportation**: Helps logistics companies estimate freight charges based on distance, weight, and demand fluctuations.
    - **📊 Supply Chain Management**: Assists businesses in forecasting shipping expenses and optimizing supply chain operations.
    - **🛳️ International Trade & Export**: Predicts shipping costs for cross-border trade, considering tariffs, fuel prices, and customs fees.
    - **🏬 Warehousing & Distribution**: Aids warehouse operators in selecting cost-efficient carriers and improving delivery speed.
    - **💰 Financial & Budget Planning**: Enables businesses to set accurate logistics budgets and reduce unexpected shipping costs.
    - **📡 Smart Shipping Platforms**: Powers AI-driven logistics platforms to automate pricing, route optimization, and cost predictions.
    """
)

# Data Visualization Part
if page == "Visualization":
    st.title("Data Visualization")

    Material_count = df['Material'].value_counts()
    plt.figure(figsize=(12,6))
    plt.title("types of material")
    plt.pie(Material_count,autopct="%1.1f%%",labels=Material_count.index)
    st.pyplot(plt)

    International_count = df['International'].value_counts()
    plt.figure(figsize=(12,6))
    plt.title("International category")
    plt.pie(International_count,autopct="%1.1f%%",labels=International_count.index)
    st.pyplot(plt)

    plt.figure(figsize=(12,6))
    sns.countplot(x = 'International',hue = 'Transport',data =df,palette='viridis')
    plt.title("International-Transport")
    plt.xlabel("International")
    plt.ylabel("Transport")
    plt.xticks(rotation =90)
    st.pyplot(plt)

    plt.figure(figsize=(12,6))
    sns.countplot(x ='Width',hue ='Cost',data =df,palette='viridis')
    plt.title("width wise product cost")
    plt.xlabel("width")
    plt.ylabel("cost")
    plt.xticks(rotation =90)
    st.pyplot(plt)

    plt.figure(figsize=(12,6))
    sns.barplot(x ='Material',hue ='Cost',data =df,palette='viridis')
    plt.title("cost based on Material")
    plt.xlabel("Material")
    plt.ylabel("cost")
    plt.xticks(rotation =90)
    st.pyplot(plt)



unique_Material = np.array(['Brass' ,'Clay', 'Aluminium', 'Wood' ,'Marble', 'Bronze', 'Stone'])
unique_Material = np.array(unique_Material).reshape(-1,1)

unique_international = np.array(['Yes' ,'No'])
unique_international = np.array(unique_international).reshape(-1,1)

unique_transport = np.array(['Airways' ,'Roadways' , 'Waterways'])
unique_transport = np.array(unique_transport).reshape(-1,1)

unique_fragile = np.array(['No' ,'Yes'])
unique_fragile = np.array(unique_fragile).reshape(-1,1)

unique_customer_info = np.array(['Working Class' ,'Wealthy'])
unique_customer_info = np.array(unique_customer_info).reshape(-1,1)

# label encoding
from sklearn.preprocessing import LabelEncoder
Material_encode = LabelEncoder()
International_encode = LabelEncoder()
Transport_encode = LabelEncoder()
Fragile_encode = LabelEncoder()
Customer_Information_encode = LabelEncoder()

Material_encode.fit(unique_Material)
International_encode.fit(unique_international)
Transport_encode.fit(unique_transport)
Fragile_encode.fit(unique_fragile)
Customer_Information_encode.fit(unique_customer_info)


def predict_shipment(features):
    features = np.array(features).reshape(1,-1)
    pred = ship.predict(features)
    return pred

# st.title("Shipment Price Prediction")

# prediction part
if page == "Prediction":
    st.title("Shipment Price Prediction")
    Artist_Reputation = st.number_input("Artist Reputation",min_value=0.0,max_value=1.0,step=0.1)
    Width = st.number_input("Width",min_value=1,max_value=50,step=1)
    Material = st.selectbox("Material",['Brass' ,'Clay', 'Aluminium', 'Wood' ,'Marble', 'Bronze', 'Stone'])
    International = st.selectbox("International",['Yes' ,'No'])
    Transport = st.selectbox("Transport",['Airways' ,'Roadways' , 'Waterways'])
    Fragile = st.selectbox("Fragile",['No' ,'Yes'])
    Customer_Information = st.selectbox("Customer_Information",['Working Class' ,'Wealthy'])
    Scheduled_year = st.number_input('Scheduled_year',min_value=2015,max_value=2019,step=1)
    Scheduled_month = st.number_input("Scheduled_month",min_value=1,max_value=12,step=1)
    Delivery_year = st.number_input("Delivery_year",min_value=2014,max_value=2019,step=1)
    Delivery_month = st.number_input("Delivery_month",min_value=1,max_value=12,step=1)

    encoded_material = Material_encode.transform([Material])
    encoded_international = International_encode.transform([International])
    encoded_transport = Transport_encode.transform([Transport])
    encoded_fragile = Fragile_encode.transform([Fragile])
    encoded_Customer_Information = Customer_Information_encode.transform([Customer_Information])

    features = [
        Artist_Reputation,
        Width,
        encoded_material.item(),
        encoded_international.item(),
        encoded_transport.item(),
        encoded_fragile.item(),
        encoded_Customer_Information.item(),
        Scheduled_year,
        Scheduled_month,
        Delivery_year,
        Delivery_month
    ]

    if st.button("Predict"):
        result = predict_shipment(features)
        st.subheader(f"Predicted Shipment Price: ${result[0]:,.2f}")

