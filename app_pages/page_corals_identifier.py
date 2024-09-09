import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image
                                                    )

def page_corals_identifier_body():
    st.info(
        f"* The client is interested in telling whether a given corals is identified as 'Bleached', 'Dead' "
        f"or 'Healthy'."
        )

    st.write(
        f"* You can download a set of images in different states of health for live prediction. "
        f"You can download the images from [here](https://www.kaggle.com/datasets/sonainjamil/bhd-corals)."
        )

    st.write("---")

    images_buffer = st.file_uploader('Upload images of corals. You may select more than one.',
                                        type=['png', 'jpeg', 'jpg', 'tiff', 'bmp', 'gif'], accept_multiple_files=True)
   
    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f"Corals snapshot: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v4'
            resized_img = resize_input_image(img=img_pil, version=version)
            load_model_and_predict(resized_img, version=version)            
        
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)


