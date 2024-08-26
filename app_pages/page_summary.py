import streamlit as st

def business_case() -> None:

    st.write("## Project Summary and business case")

    st.info(
        f"* Coral bleaching is the process when corals become white due to loss of symbiotic algae and photosynthetic pigments. This loss of pigment can be caused by various stressors, such as changes in ocean temperature (due to Global Warming), light, or nutrients. Bleaching occurs when coral polyps expel the zooxanthellae (dinoflagellates that are commonly referred to as algae) that live inside their tissue, causing the coral to turn white.\n"
        f"* The zooxanthellae are photosynthetic, and as the water temperature rises, they begin to produce reactive oxygen species. This is toxic to the coral, so the coral expels the zooxanthellae. Since the zooxanthellae produce the majority of coral colouration, the coral tissue becomes transparent, revealing the coral skeleton made of calcium carbonate. Most bleached corals appear bright white, but some are blue, yellow, or pink due to pigment proteins in the coral.\n"
        f"* The leading cause of coral bleaching is rising ocean temperatures due to climate change caused by anthropogenic activities. A temperature about 1 °C (or 2 °F) above average can cause bleaching.\n"
        f"* According to the United Nations Environment Programme, between 2014 and 2016, the longest recorded global bleaching events killed coral on an unprecedented scale. In 2016, bleaching of coral on the Great Barrier Reef killed 29 to 50 percent of the reef's coral. In 2017, the bleaching extended into the central region of the reef. The average interval between bleaching events has halved between 1980 and 2016, [Wikipedia article](https://en.wikipedia.org/wiki/Coral_bleaching).\n"        
        )

    st.write("## Dataset")

    st.info(
        f"The available dataset contains 1582 images of healthy, bleached and death corals. The dataset is downloadable from [Kaggle](https://www.kaggle.com/datasets/sonainjamil/bhd-corals).\n"
        )

    st.write(
        f"* Additional information: [Project README file](https://github.com/.../README.md).\n")

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in having a study to differentiate 'healthy', 'bleached' and 'dead' corals'\n"        
        f"* 2 - The client is interested to tell whether a given segment/part of coral colony is healthy, suffers from bleaching phenomenon or dead based on the analysis of coral images."
        )