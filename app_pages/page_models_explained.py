import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
# import tempfile
import io
import matplotlib.pyplot as plt
from PIL import Image



def page_models_explained_body():
    st.success("### Summary and structure of models used in the Project")
    version = 'v4'
    model_1 = load_model(f"outputs/{version}/model_1.h5")
    model_2 = load_model(f"outputs/{version}/model_2.h5")
    model_3 = load_model(f"outputs/{version}/model_3.h5")    
    
    st.info("#### Model_1")
    
    stringlist_1 = []
    model_1.summary(print_fn=lambda x: stringlist_1.append(x))
    summary_string_1 = "\n".join(stringlist_1)
    st.text(summary_string_1)    
        
    st.write("---")
    
    st.info("#### Model_2")
    
    stringlist_2 = []
    model_2.summary(print_fn=lambda x: stringlist_2.append(x))
    summary_string_2 = "\n".join(stringlist_2)
    st.text(summary_string_2)   
    
    
    st.write("---")
    
    st.info("#### Model_3")
    
    stringlist_3 = []
    model_3.summary(print_fn=lambda x: stringlist_3.append(x))
    summary_string_3 = "\n".join(stringlist_3)
    st.text(summary_string_3)