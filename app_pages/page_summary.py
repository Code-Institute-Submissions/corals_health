import streamlit as st


def business_case() -> None:
    """
    Business case, business objective and dataset summary
    """
    st.write("## Project Summary and Business Case")

    st.image('assets/images/coral-reef-title-image.jpg', use_column_width=True)

    st.markdown(
        '[Image source](https://www.anses.fr/en/content/coral-reefs-french-'
        'overseas-territories)')

    st.info(
        "* Coral bleaching is the process when corals become\n"
        "white due to loss of symbiotic algae and photosynthetic\n"
        "pigments. This loss of pigment can be caused by various\n"
        "stressors, such as changes in ocean temperature (due to\n"
        "Global Warming), light, or nutrients. Bleaching occurs\n"
        "when coral polyps expel the zooxanthellae (dinoflagellates\n"
        "that are commonly referred to as algae) that live inside\n"
        "their tissue, causing the coral to turn white.\n"
        "* The zooxanthellae are photosynthetic, and as the water\n"
        "temperature rises, they begin to produce reactive oxygen\n"
        "species. This is toxic to the coral, so the coral expels the\n"
        "zooxanthellae. Since the zooxanthellae produce the majority of\n"
        "coral colouration, the coral tissue becomes transparent, revealing\n"
        "the coral skeleton made of calcium carbonate. Most bleached corals\n"
        "appear bright white, but some are blue, yellow, or pink due to\n"
        "pigment proteins in the coral.\n"
        "* The leading cause of coral bleaching is rising ocean temperatures\n"
        "due to climate change caused by anthropogenic activities.\n"
        "A temperature about 1 °C (or 2 °F) above average can cause\n"
        "bleaching."
        "* According to the United Nations Environment Programme, between\n"
        "2014 and 2016, the longest recorded global bleaching events killed\n"
        "coral on an unprecedented scale. In 2016, bleaching of coral on\n"
        "the Great Barrier Reef killed 29 to 50 percent of the reef's coral.\n"
        "In 2017, the bleaching extended into the central region of the\n"
        "reef. The average interval between bleaching events has halved\n"
        "between 1980 and 2016,\n"
        "[Wikipedia article](https://en.wikipedia.org/wiki/Coral_bleaching).\n"
        )

    st.success(
        "**The project has two business requirements:**\n"
        "* 1 - The client is interested in having the capability to compare\n"
        "average images obtained for 'healthy', 'bleached' and 'dead' corals\n"
        "and check if these groups can be visually unambiguously\n"
        "categorised.\n"
        "* 2 - To answer (by using a trained ML model) whether an uploaded\n"
        "(previously unseen) image was taken of a 'Healthy', 'Bleached' or\n"
        "'Dead' coral."
        )

    st.write("## Dataset")

    st.info(
        "The available dataset contains 1582 images of healthy, bleached\n"
        "and dead corals. The dataset is downloadable from\n"
        "[Kaggle](https://www.kaggle.com/datasets/sonainjamil/bhd-corals)."
        )

    st.write(
        "* Additional information:\n"
        "[README](https://github.com/DrSYakovlev/corals_health).")
