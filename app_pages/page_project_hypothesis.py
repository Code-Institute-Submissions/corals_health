import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis")

    st.success(
        f"#### Healthy Corals:\n"
        f"* Typically vibrant in color due to the presence of symbiotic algae. Colors can range from browns and greens to bright blues, reds, and yellows, depending on the species.\n"
        f"#### Bleached Corals:\n"
        f"* Appear white or pale because they have expelled their symbiotic algae. The coral's skeleton becomes visible through their translucent tissues.\n"
        f"#### Dead Corals:\n"
        f"* Often appear dull and covered in algae or overgrown by other organisms. The structure may begin to erode and break apart. They do not regain color and look more like bare rock."
    )
