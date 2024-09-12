import requests
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file

def download_model(url, local_path):
    """
    Helper function to download a model file from a URL and save it locally.
    """
    # Ensure the target folder exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Get the model file
    response = requests.get(url, allow_redirects=True)
    with open(local_path, 'wb') as file:
        file.write(response.content)
    print("Downloaded model to:", local_path)

def resize_input_image(img, version):
    """
    Reshape image to average image size
    """
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    my_image = np.expand_dims(img_resized, axis=0) / 255

    return my_image

def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images and plot model comparison
    """
    # Ensure to use the raw GitHub URLs for direct file access
    # model_1_url = 'https://github.com/DrSYakovlev/corals_health/raw/main/outputs/v4/model_1.h5'
    # model_2_url = 'https://github.com/DrSYakovlev/corals_health/raw/main/outputs/v4/model_2.h5'
    model_3_url = 'https://github.com/DrSYakovlev/corals_health/raw/main/outputs/v4/model_3.h5'

    # Local paths where the models will be saved
    # model_1_path = 'saved_models/model_1.h5'
    # model_2_path = 'saved_models/model_2.h5'
    model_3_path = 'saved_models/model_3.h5'

    # Download models if they don't exist locally
    # if not os.path.exists(model_1_path):
    #     download_model(model_1_url, model_1_path)
    # if not os.path.exists(model_2_path):
    #     download_model(model_2_url, model_2_path)
    if not os.path.exists(model_3_path):
        download_model(model_3_url, model_3_path)

    # Load the models
    # model_1 = load_model(model_1_path)
    # model_2 = load_model(model_2_path)
    model_3 = load_model(model_3_path)

    # Perform predictions
    # pred_proba_model_1 = model_1.predict(my_image)[0]
    # pred_proba_model_2 = model_2.predict(my_image)[0]
    pred_proba_model_3 = model_3.predict(my_image)[0]    
    
    # Mapping predictions to target classes
    target_map = {v: k for k, v in {'Bleached': 0, 'Dead': 1, 'Healthy': 2}.items()}
    # prob_per_class_model_1 = pd.DataFrame(data=[pred_proba_model_1], columns=target_map.values())
    # prob_per_class_model_2 = pd.DataFrame(data=[pred_proba_model_2], columns=target_map.values())
    prob_per_class_model_3 = pd.DataFrame(data=[pred_proba_model_3], columns=target_map.values())

    combined_df = pd.concat([
    #    prob_per_class_model_1.assign(Model='Model 1'),
    #    prob_per_class_model_2.assign(Model='Model 2'),
        prob_per_class_model_3.assign(Model='Model 3')
    ]).melt(id_vars=['Model'], var_name='Results', value_name='Probability')

    # Plot the bar chart, grouped by model:
    """
    fig = px.bar(
        combined_df,
        x='Model',
        y='Probability',
        color='Results',
        barmode='group',
        range_y=[0, 1],
        width=600,
        height=400,
        template='seaborn',
        title="Probability by Model"
    )    
    st.plotly_chart(fig)
    
    """ 

    # Plot the bar chart, grouped by Coral Health State (Results):
    fig = px.bar(
        combined_df,
        x='Results',
        y='Probability',
        # color='Model',
        barmode='group',
        range_y=[0, 1],
        width=600,
        height=400,
        template='seaborn',
        title="Probability by Coral Health State"
    )
    st.plotly_chart(fig)

