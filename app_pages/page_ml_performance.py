import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from src.machine_learning.evaluate_clf import *


def page_ml_performance_metrics_body():
    """
    Function to show model training
    and label distribution in dataset
    """
    version = 'v4'
    st.write("### Train, Validation and Test Set: Labels Frequencies")
    labels_distribution = plt.imread(
        f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on\n'
             'Train, Validation and Test Sets')
    st.write("---")
    st.write("### Model History*")
    st.info("The **accuracy** and **loss** graphs below provide\n"
            "visualisation of the ML learning cycle.")
    col1, col2 = st.beta_columns(2)
    with col1:
        model_3_acc = plt.imread(f"outputs/{version}/model_3_training_acc.png")
        st.image(model_3_acc, caption='Model_3 Training Accuracy')
    with col2:
        model_3_loss = plt.imread(
            f"outputs/{version}/model_3_training_losses.png")
        st.image(model_3_loss, caption='Model_3 Training Losses')
    st.write("The model is nor overfitting neither underfitting.")
    st.write("### Generalised Performance on Test Set (Model_3)")
    st.dataframe(pd.DataFrame(load_test_evaluation_model_3(version),
                              index=['Loss', 'Accuracy']))
    st.success("The general accuracy of the model is **85 %**")
    st.write("---")
    st.write("###### *During model development, several models\n"
             "###### were trained and tested.")
    st.write("###### **Model_3** showed the best performance\n"
             "###### during evaluation.")
    st.write("###### For more information see\n"
             "###### [README](https://github.com/DrSYakovlev/corals_health).")
