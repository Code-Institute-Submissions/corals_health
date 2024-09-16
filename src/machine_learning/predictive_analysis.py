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


def load_model_and_predict(my_image, image_name, version):
    """
    Load and perform ML prediction over live images and plot model comparison
    """
    # Provide here model_1 and model_2 url for multimodel
    model_3_url = (
        'https://github.com/DrSYakovlev/corals_health/raw/main/'
        'outputs/v4/model_3.h5')

    # Local paths where the models will be saved
    # Provide here local path for model_1 and model_2
    # for multimodel
    model_3_path = 'saved_models/model_3.h5'

    # Download models if they don't exist locally
    if not os.path.exists(model_3_path):
        download_model(model_3_url, model_3_path)

    # Load the models
    # Load here model_1 and model_2 for multimodel
    model_3 = load_model(model_3_path)

    # Perform predictions
    # Perform here prediction on model_1 and model_2
    # for multimodel
    pred_proba_model_3 = model_3.predict(my_image)[0]

    # Mapping predictions to target classes
    target_map = {
        v: k
        for k, v in {
            'Bleached': 0,
            'Dead': 1,
            'Healthy': 2
         }.items()}

    # Add here prob_per_class_model_1 and _model_2 for multimodel
    prob_per_class_model_3 = pd.DataFrame(
        data=[pred_proba_model_3], columns=target_map.values())

    combined_df = pd.concat([
        # prob_per_class_model_1 and model_2: .assign() here for multimodel
        prob_per_class_model_3.assign(Model='Model 3')
    ]).melt(id_vars=['Model'], var_name='Results', value_name='Probability')

    fig = px.bar(
        combined_df,
        x='Results',
        y='Probability',
        # assign here: color='Model', for multi-model app
        barmode='group',
        range_y=[0, 1],
        width=600,
        height=400,
        template='seaborn',
        title="Probability by Coral Health State"
    )
    st.plotly_chart(fig)

    # Determine the most likely class or uncertainty message
    max_prob = prob_per_class_model_3.max(axis=1).values[0]
    predicted_class = prob_per_class_model_3.idxmax(axis=1).values[0]

    if max_prob > 0.5:
        st.success(f"The model says that the colony is the\n"
                   f"most likely **{predicted_class}**\n"
                   "(more than 50 % probability).")
    else:
        st.success(f"The model is **not sure** (the probability for each\n"
                   "group is below 50 %). Please,\n"
                   "try another image of the colony.")


def final_report(image_name, my_image):
    """
    Genarate final report
    """
    model_3_url = (
        'https://github.com/DrSYakovlev/corals_health/raw/main/'
        'outputs/v4/model_3.h5')

    model_3_path = 'saved_models/model_3.h5'
    if not os.path.exists(model_3_path):
        download_model(model_3_url, model_3_path)
    model_3 = load_model(model_3_path)
    pred_proba_model_3 = model_3.predict(my_image)[0]
    target_map = {
        v: k
        for k, v in {
            'Bleached': 0,
            'Dead': 1,
            'Healthy': 2
         }.items()}
    prob_per_class_model_3 = pd.DataFrame(
        data=[pred_proba_model_3], columns=target_map.values())
    max_prob = prob_per_class_model_3.max(axis=1).values[0]
    predicted_class = prob_per_class_model_3.idxmax(axis=1).values[0]
    if max_prob > 0.5:
        result = f"{predicted_class}"
    else:
        result = "not sure"
    report_df = pd.DataFrame({
        'id': [image_name],
        '---': [result]
    })
    st.dataframe(report_df)
