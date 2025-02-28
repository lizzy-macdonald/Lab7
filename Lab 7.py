import pandas as pd
# Read the dataset from the local file in the GitHub repository
df = pd.read_excel(’AmesHousing.xlsx’)

# This app is for educaiton demonstration purpose that teaches students how to develop and deploy an interactive web based engineering application app.
# Data source: uc irvine machine learning repository 

# Load libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import openpyxl
import xlrd

# Load the data
df = pd.read_excel(’AmesHousing.xlsx’)

# Clean column names
data.columns = [col.split('(')[0].strip() for col in data.columns]
data.rename(columns={'Yr Sold': 'SalePrice'}, inplace=True)

# Assuming no missing values, split the data into features and target
X = data.drop(columns=['Strength'])
y = data['Strength']

# Train a Multiple Regression Model 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

# Create the streamlit web-based app

# Title of the app
st.title('Housing  Price Prediction')

# Sidebar for user inputs
st.sidebar.header('Input Parameters')

def user_input_features():
    Bsmt Full Bath = st.sidebar.slider('Basement Full Bath', 0, 1, 2)
    Bsmt Half Bath = st.sidebar.slider('Basement Half Bath ', 0, 1, 2)
    Full Bath = st.sidebar.slider('Full Bath', 0, 1, 2)
    Half Bath = st.sidebar.slider('Half Bath', 0, 1, 2)
    Bedroom = st.sidebar.slider('Bedroom 1, 2, 3, 4)
    Kitchen AbvGr = st.sidebar.slider('Kitchen Above Ground', 0, 1)
    
    data = {
        'Basement Full Bath': Bsmt Full Bath,
        'Basement Half Bath': Bsmt Half Bath,
        'Full Bath': Full Bath,
        'Half Bath': Half Bath,
        'Bedroom': Bedroom,
        'Kitchen Above Ground': Kitchen AbvGr,
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input Parameters')
st.write(input_df)

# Predict the compressive strength
prediction = model.predict(input_df)

# Display the prediction
st.subheader('Predicted Concrete Compressive Strength (MPa)')
st.write(prediction[0])
