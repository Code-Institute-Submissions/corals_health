import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_corals_visualizer_body():
    st.write("### Corals visualisation")
    st.info(
        f"* The client is interested in having a capability to compare "
        f"average images obtained for 'healthy', 'bleached' and 'dead' corals "
        f"and check if these groups can be visually unambiguously categorised.")
    
    version = 'v4'
    if st.checkbox("Difference between averages for different groups"):
      
      avg_healthy_bleached = plt.imread(f"outputs/{version}/avg_diff_healthy-bleached.png")
      avg_healthy_dead = plt.imread(f"outputs/{version}/avg_diff_healthy_dead.png")
      avg_bleached_dead = plt.imread(f"outputs/{version}/avg_diff_bleached_dead.png")      

      st.image(avg_healthy_bleached, caption='Difference between averages for Healthy and Bleached corals')
      st.image(avg_healthy_dead, caption='Difference between averages for Healthy and Dead dorals')
      st.image(avg_bleached_dead, caption='Difference between averages for Bleached and Dead corals')
      st.write("---")
      st.warning(
        f"* The observation is that on average there is very small difference\n"
        f"between snapshots taken for healthy, bleached and dead corals.\n " 
        f"Minor (some) difference is seen for pair 'healthy' - 'dead'.\n"
        f"No visual difference was detedted for pairs 'healthy' - 'bleached' "
        f"and 'bleached' - 'dead'."
      )
    """
    if st.checkbox("Differences between average parasitised and average uninfected cells"):
          diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

          st.warning(
            f"* We notice this study didn't show "
            f"patterns where we could intuitively differentiate one from another.")
          st.image(diff_between_avgs, caption='Difference between average images')
    """

    if st.checkbox("Image Montage"): 
      st.write("* To refresh the montage, click on the 'Create Montage' button")
      my_data_dir = os.path.join('inputs', 'corals-dataset', 'Dataset')
      labels = os.listdir(os.path.join(my_data_dir, 'validation'))
      label_to_display = st.selectbox(label="Select label", options=labels, index=0)
      if st.button("Create Montage"):      
        image_montage(os.path.join(my_data_dir, 'validation'),
                      label_to_display=label_to_display,
                      nrows=5, ncols=3, figsize=(10,25))
      st.write("---")

def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
  sns.set_style("white")
  labels = os.listdir(dir_path)

  # subset the class you are interested to display
  if label_to_display in labels:

    # checks if your montage space is greater than subset size
    # how many images in that folder
    images_list = os.listdir(os.path.join(dir_path, label_to_display))
    if nrows * ncols < len(images_list):
      img_idx = random.sample(images_list, nrows * ncols)
    else:
      print(
          f"Decrease nrows or ncols to create your montage. \n"
          f"There are {len(images_list)} in your subset. "
          f"You requested a montage with {nrows * ncols} spaces")
      return
    

    # create list of axes indices based on nrows and ncols
    list_rows= range(0, nrows)
    list_cols= range(0, ncols)
    plot_idx = list(itertools.product(list_rows,list_cols))


    # create a Figure and display images
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
    for x in range(0,nrows*ncols):
      img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
      img_shape = img.shape
      axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
      axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
      axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
      axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()
    
    st.pyplot(fig=fig)
    # plt.show()


  else:
    print("The label you selected doesn't exist.")
    print(f"The existing options are: {labels}")