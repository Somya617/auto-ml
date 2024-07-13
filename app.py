import streamlit as st
import pandas as pd
import os
import requests
from pycaret.classification import setup, compare_models, pull, save_model
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.datasets import load_iris


# Sidebar navigation
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoMobiusML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download", "API Call"])
    st.info("This project application helps you build and explore your data.")

def load_file(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, index_col=None)
            df.to_excel("sourcedata.xlsx", index=None)
        else:
            df = pd.read_csv(file, index_col=None, encoding='latin1')
            df.to_csv("sourcedata.csv", index=None)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if os.path.exists("sourcedata.xlsx"):
    df = pd.read_excel("sourcedata.xlsx", index_col=None)
elif os.path.exists("sourcedata.csv"):
    try:
        df = pd.read_csv("sourcedata.csv", index_col=None, encoding='latin1')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = None

# Upload Dataset
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=["csv", "xlsx"])
    if file:
        df = load_file(file)
        if df is not None:
            st.dataframe(df)

# Profiling
if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    if 'df' in locals() and df is not None:
        profile_report = ProfileReport(df, title="Pandas Profiling Report")
        st_profile_report(profile_report)
    else:
        st.error("Please upload a dataset first.")

# Modelling
if choice == "Modelling":
    if 'df' in locals() and df is not None:
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            setup(df, target=chosen_target, verbose=False)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
    else:
        st.error("Please upload a dataset first.")

# Download Model
if choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.error("No model available for download. Please run the modelling step first.")

# API Call
if choice == "API Call":
    st.title("API Call to Model Endpoint")
    
    # Create and save the Iris dataset
    iris = load_iris()
    df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target  # Add the target column (species)
    excel_file = 'iris_dataset.csv'
    df_iris.to_csv(excel_file, index=False)
    #df_iris.to_excel(excel_file, index=False)
    st.write(f"Iris dataset saved to {excel_file}")
    
    # Define the API endpoint URL
    url = 'http://127.0.0.1:5000/best_model'
    
    # Data to be sent to the API
    data = {'model_type': 'regression', 'testing': 'true'}
    
    # Send the file to the API
    with open(excel_file, 'rb') as file:
        files = {'file': file} 
        #data["file"] = file.read()
        print(excel_file)
        try:
            response = requests.post(url, data=data, files=files)
            response.raise_for_status()  # Raise an error for bad status codes
            response_data = response.json()
            st.write(response_data)
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except ValueError as e:
            st.error(f"JSON decoding failed: {e}")