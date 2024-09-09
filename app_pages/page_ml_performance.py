import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from src.machine_learning.evaluate_clf import *


def page_ml_performance_metrics_body():
    version = 'v4'

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")


    st.write("### Model_1 History")
    col1, col2 = st.beta_columns(2)
    with col1: 
        model_1_acc = plt.imread(f"outputs/{version}/model_1_training_acc.png")
        st.image(model_1_acc, caption='Model_1 Training Accuracy')
    with col2:
        model_1_loss = plt.imread(f"outputs/{version}/model_1_training_losses.png")
        st.image(model_1_loss, caption='Model_1 Training Losses')
    

    st.write("### Generalised Performance on Test Set (Model_1)")
    st.dataframe(pd.DataFrame(load_test_evaluation_model_1(version), index=['Loss', 'Accuracy']))
    st.write("---")
    
    
    st.write("### Model_2 History")
    col1, col2 = st.beta_columns(2)
    with col1: 
        model_2_acc = plt.imread(f"outputs/{version}/model_2_training_acc.png")
        st.image(model_2_acc, caption='Model_2 Training Accuracy')
    with col2:
        model_2_loss = plt.imread(f"outputs/{version}/model_2_training_losses.png")
        st.image(model_2_loss, caption='Model_2 Training Losses')
    st.write("### Generalised Performance on Test Set (Model_2)")
    st.dataframe(pd.DataFrame(load_test_evaluation_model_2(version), index=['Loss', 'Accuracy']))
    st.write("---")
    
    
    st.write("### Model_3 History")
    col1, col2 = st.beta_columns(2)
    with col1: 
        model_3_acc = plt.imread(f"outputs/{version}/model_3_training_acc.png")
        st.image(model_3_acc, caption='Model_3 Training Accuracy')
    with col2:
        model_3_loss = plt.imread(f"outputs/{version}/model_3_training_losses.png")
        st.image(model_3_loss, caption='Model_3 Training Losses')
    st.write("### Generalised Performance on Test Set (Model_3)")
    st.dataframe(pd.DataFrame(load_test_evaluation_model_3(version), index=['Loss', 'Accuracy']))
    
