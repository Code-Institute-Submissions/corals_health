import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file


def resize_input_image(img, version):
    """
    Reshape image to average image size
    """
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    my_image = np.expand_dims(img_resized, axis=0)/255

    return my_image


def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images
    """

    target_map = {v: k for k, v in {'Bleached': 0, 'Dead': 1, 'Healthy': 2}.items()}
    
    model_1 = load_model(f"outputs/{version}/model_1.h5")
    # model_2 = load_model(f"outputs/{version}/model_2.h5")
    # model_3 = load_model(f"outputs/{version}/model_3.h5")

    pred_proba_model_1 = model_1.predict(my_image)[0]
    # pred_proba_model_2 = model_2.predict(my_image)[0]
    # pred_proba_model_3 = model_3.predict(my_image)[0]    
    
    prob_per_class_model_1 = pd.DataFrame(data=pred_proba_model_1, columns=['Probability'])
    # prob_per_class_model_2 = pd.DataFrame(data=pred_proba_model_2, columns=['Probability'])
    # prob_per_class_model_3 = pd.DataFrame(data=pred_proba_model_3, columns=['Probability'])    
    
    prob_per_class_model_1 = prob_per_class_model_1.round(3)
    # prob_per_class_model_2 = prob_per_class_model_2.round(3)
    # prob_per_class_model_3 = prob_per_class_model_3.round(3)
    
    prob_per_class_model_1['Results'] = target_map.values()
    # prob_per_class_model_2['Results'] = target_map.values()
    # prob_per_class_model_3['Results'] = target_map.values()
    
    combined_df = pd.concat([
    prob_per_class_model_1.assign(Model='Model 1')#,
    # prob_per_class_model_2.assign(Model='Model 2'),
    # prob_per_class_model_3.assign(Model='Model 3')
])
    
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
    # Plot the bar chart, grouped by Coral Health State (Results)
    fig = px.bar(
    combined_df,
    x='Results',
    y='Probability',
    color='Model',
    barmode='group',
    range_y=[0, 1],
    width=600,
    height=400,
    template='seaborn',
    title="Probability by Coral Health State"
)

    fig.show()
    """