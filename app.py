# Import streamlit and page functions
import streamlit as st
from app_pages.multipage import MultiPage
from app_pages.page_summary import business_case
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_corals_visualizer import page_corals_visualizer_body
from app_pages.page_corals_identifier import page_corals_identifier_body

# from app_pages.page_ml_performance import page_ml_performance_metrics

app = MultiPage(app_name="Corals Health")  # Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("Business case", business_case)
app.add_page("Project Hypothesis", page_project_hypothesis_body)

app.add_page("Corals Visualiser", page_corals_visualizer_body)
app.add_page("Corals State Identifier", page_corals_identifier_body)

# app.add_page("ML Performance Metrics", page_ml_performance_metrics)

app.run()  # Run the app
