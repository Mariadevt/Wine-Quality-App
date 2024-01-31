import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#--------STREAMLIT APP------#

st.title("Wine Quality Prediction App")

#Loading the dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

#Load the image

st.image('wine quality.jpeg')

# Add the caption

st.caption("Wine Quality Dataset Explorer and RandomForest Classifier Model")

# Add the subheader

st.subheader("Dataset", divider='violet')

#To view the dataset

st.write(data.head())

#EDA subheader

st.subheader("Explaratory Data Analysis(EDA)", divider='violet')

#Buttons for the EDA

if st.button("Column Names"):
    st.write("Dataset Columns", data.columns)

if st.button("Missing Values"):
    st.write("Sum of Missing Values in each column", data.isnull().sum())

# Visualisation Subheaer

st.subheader("Data Visualisation", divider='violet')


#Checkbox for the Bar Chart and Line Chart

if st.checkbox("Bar Plot of Residual Sugar Against Quality"):
    st.bar_chart(x="residual sugar", y="quality", data=data)


if st.checkbox("Line Plot of Residual Sugar Against Quality"):
      st.line_chart(x="residual sugar", y="quality" ,data=data)
      
#Prepare the data

X = data.drop("quality", axis=1)
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

#Create and fit our model

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#User Input

st.sidebar.header('Slide your values')

fixed_acidity = st.sidebar.slider("Fixed Acidity", (data ["fixed acidity"]).min(),(data ["fixed acidity"]).max(),(data ["fixed acidity"]).mean())
volatile_acidity = st.sidebar.slider("Volatile Acidity", (data ["volatile acidity"]).min(),(data ["volatile acidity"]).max(),(data ["volatile acidity"]).mean())
citric_acid = st.sidebar.slider("Citric Acid", (data ["citric acid"]).min(),(data ["citric acid"]).max(),(data ["citric acid"]).mean())
residual_sugar = st.sidebar.slider("Residual Sugar", (data ["residual sugar"]).min(),(data ["residual sugar"]).max(),(data ["residual sugar"]).mean())
chlorides = st.sidebar.slider("Chlorides", (data ["chlorides"]).min(),(data ["chlorides"]).max(),(data ["chlorides"]).mean())
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide ", (data ["free sulfur dioxide"]).min(),(data ["free sulfur dioxide"]).max(),(data ["free sulfur dioxide"]).mean())
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide ", (data ["total sulfur dioxide"]).min(),(data ["total sulfur dioxide"]).max(),(data ["total sulfur dioxide"]).mean())
density = st.sidebar.slider("Density", (data ["density"]).min(),(data ["density"]).max(),(data ["density"]).mean())
ph = st.sidebar.slider("pH", (data ["pH"]).min(),(data ["pH"]).max(),(data ["pH"]).mean())
sulphates = st.sidebar.slider("Sulphates", (data ["sulphates"]).min(),(data ["sulphates"]).max(),(data ["sulphates"]).mean())
alcohol = st.sidebar.slider("Alcohol", (data ["alcohol"]).min(),(data ["alcohol"]).max(),(data ["alcohol"]).mean())


#Predict Button

if st.sidebar.button("Predict"):
    
    #Create a dataframe for the user inputs
    
    user_input = pd.DataFrame(
        {
            'fixed acidity':[fixed_acidity],
            'volatile acidity':[volatile_acidity],
            "citric acid": [citric_acid],
            "residual sugar": [residual_sugar],
            "chlorides":[chlorides],
            "free sulfur dioxide":[free_sulfur_dioxide],
            "total sulfur dioxide":[total_sulfur_dioxide],
            "density": [density],
            "pH":[ph],
            "sulphates":[sulphates],
            "alcohol":[alcohol]
        }
    )
    
    #Prediction of wine quality
    
    prediction = rf.predict(user_input)
    
    #Display of the prediction
    st.sidebar.subheader("Prediction")
    st.sidebar.write(f"From the information provided the wine qulaity is {prediction[0]}")
    