import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    final_report
                                                    )


def page_corals_identifier_body():
    st.info(
        "* The client is interested in telling whether a\n"
        "given corals is identified as 'Bleached', 'Dead'\n"
        "or 'Healthy'."
        )

    st.write(
        "* You can download a set of images in different\n"
        "states of health for live prediction. You can\n"
        "download the images from\n"
        "[here](https://www.kaggle.com/datasets/sonainjamil/bhd-corals)."
        )

    st.write("---")

    images_buffer = st.file_uploader(
        'Upload images of corals. You may select more than one.',
        type=['png', 'jpeg', 'jpg', 'tiff', 'bmp', 'gif'],
        accept_multiple_files=True)

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f"Corals snapshot: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size:\n"
                     "{img_array.shape[1]}px width x\n"
                     "{img_array.shape[0]}px height")

            version = 'v4'
            resized_img = resize_input_image(img=img_pil, version=version)
            image_name = image.name
            load_model_and_predict(resized_img, image_name, version=version)

        st.write('**Report summary:**')

        for image in images_buffer:
            img_pil = (Image.open(image))
            version = 'v4'
            resized_img = resize_input_image(img=img_pil, version=version)
            image_name = image.name
            final_report(image_name, resized_img)
