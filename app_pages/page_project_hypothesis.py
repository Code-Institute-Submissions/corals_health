import streamlit as st


def page_project_hypothesis_body():
    """
    Function to show project hypothesis page
    """
    st.write("### Project Hypothesis")

    st.success(
        "#### Healthy Corals:\n"
        "* Typically vibrant in color due to the presence of symbiotic\n"
        "algae. Colors can range from browns and greens to bright blues,\n"
        "reds, and yellows, depending on the species.\n"
        "#### Bleached Corals:\n"
        "* Appear white or pale because they have expelled their symbiotic\n"
        "algae. The coral's skeleton becomes visible through their\n"
        "translucent tissues.\n"
        "#### Dead Corals:\n"
        "* Often appear dull and covered in algae or overgrown by other\n"
        "organisms. The structure may begin to erode and break apart.\n"
        "They do not regain color and look more like bare rock."
    )
