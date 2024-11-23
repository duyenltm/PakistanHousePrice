import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.title('Pakistan House Price Prediction')
st.write('---')

# Load dataset
url = 'https://drive.google.com/uc?id=1HPzLNrEIBduaatuEsf7EDWJ2_J0f4T2N'
data = pd.read_csv(url)

# Data preprocessing
data = data.drop(columns=['property_id', 'page_url', 'location_id', 'province_name', 'latitude', 'longitude', 
                          'date_added', 'agency', 'agent', 'Area Category'])
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data = data[(data['price'] > 0) & (data['baths'] > 0) & (data['bedrooms'] > 0) & (data['Area Size'] > 0)]

data['Area Size'] = data.apply(lambda row: row['Area Size'] * 272.51 if row['Area Type'] == 'Marla' 
                               else row['Area Size'] * 5445, axis=1)
data.drop(['Area Type'], axis=1, inplace=True)

# Outlier removal
X_data = data.select_dtypes(include=['float64', 'int64'])
Q1 = X_data.quantile(0.25)
Q3 = X_data.quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (X_data < (Q1 - 1.5 * IQR)) | (X_data > (Q3 + 1.5 * IQR))
data = data[~outlier_condition.any(axis=1)]

# Scaling
scaler = MinMaxScaler()
data[['price_scaled', 'area_scaled', 'baths_scaled', 'bedrooms_scaled']] = scaler.fit_transform(
    data[['price', 'Area Size', 'baths', 'bedrooms']])
data = data.drop(columns=['price', 'Area Size', 'baths', 'bedrooms'])

# Encoding
le = LabelEncoder()
data['purpose'] = le.fit_transform(data['purpose'])
data['property_type'] = le.fit_transform(data['property_type'])
data['city'] = le.fit_transform(data['city'])
data['location'] = le.fit_transform(data['location'])

# Splitting data
X = data.drop(['price_scaled'], axis=1)
y = data['price_scaled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Input form
def user_input():
    with st.form("input_form"):
        st.header("Enter House Details")
        property_type = st.selectbox('Property Type', data['property_type'].unique())
        location = st.selectbox('Location', data['location'].unique())
        city = st.selectbox('City', data['city'].unique())
        baths = st.slider('Baths', 1, 7)
        purpose = st.selectbox('Purpose', data['purpose'].unique())
        bedrooms = st.slider('Bedrooms', 1, 7)
        area_size = st.number_input('Area Size', 1, 1000)
        area = area_size * 272.51
        submitted = st.form_submit_button("Submit")
        if submitted:
            input_data = {'property_type': property_type,
                          'location': location,
                          'city': city,
                          'purpose': purpose,
                          'area_scaled': area,
                          'baths_scaled': baths,
                          'bedrooms_scaled': bedrooms}
            return pd.DataFrame([input_data])
        return None

df = user_input()

# Prediction
if df is not None:
    df_scaled = scaler.transform(df)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    joblib.dump(model, 'best_model.pkl')
    best_model = joblib.load('best_model.pkl')
    prediction = best_model.predict(df_scaled)
    original_price = scaler.inverse_transform([[prediction[0], 0, 0, 0]])[0][0]

    st.header('Predicted Price')
    st.write(f"The predicted price is: {original_price:.2f}")
    st.write('---')

    # Feature Importance
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X)
    st.header('Feature Importance')
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(plt)
