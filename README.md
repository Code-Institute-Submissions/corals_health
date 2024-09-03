# CORALS HEALTH

![Coral Reef](./assets/images/coral-reef-title-image.jpg)
[Image source](https://www.anses.fr/en/content/coral-reefs-french-overseas-territories)

#### INTRODUCTION

Welcome to the "Coral's Health" ML project. This project was instpired by several visits of the author to the Great Barrier Reef and observations of detrimantal effect of human activity on various colonies of corals.<br>
<br>
text placeholder

## Table of Contents

### [I. CRISP-DM Methodology](#crisp-dm-methodology)
* [1. Business Understanding](#business-understanding)
* [2. Data Understanding](#data-understanding)
* [3. Data Preparation](#data-preparation)
* 4. Modelling (*incl model type/algorithm selection*)
* 5. Evaluation
* 6. Deployment

### II. ML Pipeline
* 1. Data Collection
* 2. Exploratory Data Analysis (EDA)
* 3. Feature Engineering
* 4. Model Building
* 5. Model Evaluation


### [III. Jupyter Notebooks](#iii-jupyter-notebooks)
* [1. Notebook Structure](#notebook-structure)
* [2. Results and Visualisation](#results-and-visualisation)


### IV. Streamlit Dashboard
* 1. Overview
* 2. Installation and Setup
* 3. Backend Code: Features and Validation
<b>Do not forget about PEP8 Evaluation</b>
* 4. Dashboard features
* 5. User Guide

### V. Conclusions and Future Work

### VI. Dependencies and Requirements

### VII. How to Run the Project

### VIII. Acknowledgements

#### Contact information
<hr>
<hr>

### CRISP-DM Methodology

#### Business Understanding
Coral bleaching is the process when corals become white due to loss of symbiotic algae and photosynthetic pigments. This loss of pigment can be caused by various stressors, such as changes in ocean temperature (due to Global Warming), light, or nutrients. Bleaching occurs when coral polyps expel the zooxanthellae (dinoflagellates that are commonly referred to as algae) that live inside their tissue, causing the coral to turn white. The zooxanthellae are photosynthetic, and as the water temperature rises, they begin to produce reactive oxygen species. This is toxic to the coral, so the coral expels the zooxanthellae. Since the zooxanthellae produce the majority of coral colouration, the coral tissue becomes transparent, revealing the coral skeleton made of calcium carbonate. Most bleached corals appear bright white, but some are blue, yellow, or pink due to pigment proteins in the coral. The leading cause of coral bleaching is rising ocean temperatures due to climate change caused by anthropogenic activities. A temperature about 1 °C (or 2 °F) above average can cause bleaching. According to the United Nations Environment Programme, between 2014 and 2016, the longest recorded global bleaching events killed coral on an unprecedented scale. In 2016, bleaching of coral on the Great Barrier Reef killed 29 to 50 percent of the reef's coral. In 2017, the bleaching extended into the central region of the reef. The average interval between bleaching events has halved between 1980 and 2016, [Wikipedia article](https://en.wikipedia.org/wiki/Coral_bleaching)."
<br>
<br>
The project has two business requirements:<br>
<hr>
<b>1. - The client is interested in having a capability to compare average images obtained for 'healthy', 'bleached' and 'dead' corals and check if these groups can be visually unambiguously categorised.</b>
<hr>
<b>2. - Answer (by using a trained ML model) if an uploaded (previously unseen) image was taken for a 'Healthy', 'Bleached' or 'Dead' coral.</b>
<hr>
Addressing these requirements sucessfully will (a)reduce an amount of tedious/repetitive work, (b) reduce involvement of an operator/technician and therefore minimize a human error, and (c) will help with the task to a visually-impaired user. For instance, colour-blindeness can make a task of manual sorting of images quite challenging.


#### Data Understanding
Thereare quite a few datasets (available on [Kaggle](https://www.kaggle.com)) for corals study splitted in two labels: 'Healthy' and 'Bleached'. These datasets are suitable for training models for binary classification task (similar to e.g. [Malaria-Detector walkthrough](https://malaria-predictor.onrender.com/)). It was also observed that 'Bleached' corals group actually encompasses two subgroups: 'Bleached' and 'Dead'. The aim of the project and the idea of the author was to check if a ML model (or models) can be extended and trained to enable gathegorical classification of images of corals into these **three** groups ('Bleached', 'Healthy' and 'Dead') rather than just **two** ('Healthy' and 'Bleached').<br>
The data where coral images were pre-labelled according to those three categories are much more scarce. The [dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals) found for this project contains 1572 labelled images.

#### Data Preparation 
The images were preprocessed by the owner of dataset to have identical 227 x 227 dimension. Data preparation included: (i) cleaning (ii) augmentation.

* Cleaning:

1) Checking if the files which are not images are present (and deleting them):
```
import os
def remove_non_image_file(my_data_dir):
    image_extension = ('.png', '.jpg', '.jpeg')
    folders = os.listdir(my_data_dir)
    for folder in folders:
        files = os.listdir(os.path.join(my_data_dir, folder))
        i = []
        j = []
        for given_file in files:
            if not given_file.lower().endswith(image_extension):
                file_location = os.path.join(my_data_dir, folder, given_file)
                print(file_location)
                os.remove(file_location)  # remove non image file
                i.append(1)
            else:
                j.append(1)
                pass
        print(f"Folder: {folder} - has image file", len(j))
        print(f"Folder: {folder} - has non-image file", len(i))

```
2) Checking if the dataset contains graiscale images (and deleting them):
```
def remove_black_white (my_data_dir):
    folders = os.listdir(my_data_dir)
    print(folders)
    for folder in folders:
        files = os.listdir(os.path.join(my_data_dir, folder))
        for given_image in files:
            count = 0
            image = imread(os.path.join(my_data_dir, folder, given_image))
            if len(image.shape) == 3:
                pass
            else:
                os.remove(os.path.join(my_data_dir, folder, given_image))      
                count += 1
    print(f"{count} black-white images were detected and removed.")
```

3) Manual data check:<br>

During initial attempts of a model (see correspondint sections for details) training and validation it was discovered that the model behaves well and stops training according to pre-defined stopping criteria. However, the model showed quite pure generalisation when processing unseen data. This could be an indication of overfitting. This dictated the need to have a closer look at the dataset. During manual check, it was discovered that the [dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals) contains some images in wrong folders.  The dataset used in this work was assembled and preprocessed in the context of the project, published by [Jamil <em>et al.</em>](https://www.mdpi.com/2504-2289/5/4/53). Although, the [dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals) of coral images is labelled as 'Healthy', 'Bleached' and 'Dead', the work was focused on distinguishing beween 'Healthy' and 'Bleached' (binary classification task) using 'specific deep convolutional neural networks such as AlexNet, GoogLeNet, VGG-19, ResNet-50, Inception v3, and CoralNet. (c)' The subset labelled as 'Dead' was treated as 'Bleached'. Attempt to train the model to categorise the data into three groups: 'Bleached', 'Dead' and 'Healthy' (this work) resulted in poor genaralisation and overfitting. Manual inspection of the dataset revealed that some of the 'Dead' corals were labelled as 'Bleached' and the other way around. Futhermore, some of the 'Bleached' corals were marked as 'Healthy'. This image misplacement may be less critical for binary classification, but crucial for training models for more categories. Therefore, the author had to visually inspect the entire dataset and manually move some images in downloaded dataset where the misplacement was obvous, into more appropriate folders, following the [description](https://en.wikipedia.org/wiki/Coral_bleaching). It must be mentioned that apart from the described cross-contamination, the original dataset contains stranger data, such as this: ![garden stuff.](./assets/images/salad_leaf_coral.jpg)

* Dataset splitting:<br>

After data cleaning, the dataset was split into 'train', 'validadion' and 'test' subsets in the ratio 0.7/0.1/0.2. (see [data collection notebook](#iii-jupyter-notebooks-1)):
```
import os
import shutil
import random


def split_train_validation_test_images(my_data_dir, train_set_ratio, validation_set_ratio, test_set_ratio):

    if train_set_ratio + validation_set_ratio + test_set_ratio != 1.0:
        print("train_set_ratio + validation_set_ratio + test_set_ratio should sum to 1.0")
        return

    # gets classes labels
    labels = os.listdir(my_data_dir)  # it should get only the folder name
    print(labels)
    if 'test' in labels:
        pass
    else:
        # create train, test folders with classes labels sub-folder
        for folder in ['train', 'validation', 'test']:
            for label in labels:
                os.makedirs(os.path.join(my_data_dir, folder, label))

        for label in labels:

            files = os.listdir(os.path.join(my_data_dir, label))
            random.shuffle(files)

            train_set_files_qty = int(len(files) * train_set_ratio)
            validation_set_files_qty = int(len(files) * validation_set_ratio)

            count = 1
            for file_name in files:
                if count <= train_set_files_qty:
                    # move a given file to the train set
                    shutil.move(os.path.join(my_data_dir, label, file_name), os.path.join(
                                my_data_dir, 'train', label, file_name))

                elif count <= (train_set_files_qty + validation_set_files_qty):
                    # move a given file to the validation set
                    shutil.move(os.path.join( my_data_dir, label, file_name), os.path.join(
                                my_data_dir, 'validation', label, file_name))

                else:
                    # move given file to test set
                    shutil.move(os.path.join(my_data_dir, label, file_name), os.path.join(
                                my_data_dir, 'test', label, file_name))

                count += 1

            os.rmdir(os.path.join(my_data_dir,label))

```

* Augmentation:<br>

The [dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals) found for this project contains 1572 labelled images. See corresponding section in [Juoyter notebook resulst and visualisation](#results-and-visualisation) chapter for visual representation of image number distribution in various subsets and labels. The total number is very modest which is the cause of poor generalisation of trained model. In order to improve model training and generalisation when analyzing previously unseen data, augmentation process was implemented:
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
augmented_image_data = ImageDataGenerator(rotation_range=30,
                                          width_shift_range=0.20,
                                          height_shift_range=0.20,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          fill_mode='nearest',
                                          rescale=1./255
                                          )
```









*Mention: there is no clear borderline between 'Bleached' and 'Dead'*
*Mention: salad*
*Mention: The number of images is very low, but it is definitely a start and the model can be further trained using extended dataset*

The [dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals)









### II. ML Pipeline





### III. Jupyter Notebooks

#### Notebook Structure
The project is described and summarised in three notebooks:<br>
1. Data Collection
2. Visualisation
3. Models evaluation and cross comparing.

<u><em>Data collection</em></u> notebook contain the code for: (i) setting up folder structure of the project,  acquiring the dataset from [Kaggle.com](https://www.kaggle.com/datasets/sonainjamil/bhd-corals); (ii) 2-step automated data cleaning: checking for presence of files which are not images and checking if images are not black/white (this would result in code breaking during further data analysis); (iii) splitting the dataset into training, validation and test subsets in ratio 0.7/0.1/0.2 ratio, respectively.

<u><em>Visualisation</em></u> notebook answers the 1st business requirement: <b>"The client is interested in having a capability to compare average images obtained for 'healthy', 'bleached' and 'dead' corals and check if these groups can be visually unambiguously categorised</b>". It contains the code for: (i) visualisation of image data (average image size across the entire dataset); (ii) functions to load images, shapes and labels in NumPy array; (iii) plotting and saving mean and variability images for each label; (iv) plot difference images between average images in pairs 'dead'-'bleached', 'dead'-'healthy' and 'bleached'-'healthy'; (v) plot montage of random images for 'healthy', 'dead' and 'bleached' labels.

<u><em>Models evaluation and cross comparing</em></u> notebook answers the 2nd business requirement: <b>"Can we tell if an uploaded (previously unseen) image was taken for a 'Healthy', 'Bleached' or 'Dead' coral?"</b>. It contains the code for: (i)plotting the distribution of image numbers in train, validation and test data subsets for different labels in the dataset; (ii) generating and plotting augmentation data; (iii) building the models (including defining the early stop criteria); (iv) models training; (v) models performance assessment; (vi) models evaluation; (vii) prediction using the new data and plotting probability prediction for different models.

#### Results and Visualisation
This paragraphs summarises graphical outcomes of the jupyter notebooks described above.

* Dataset folder structure after splitting the dataset into the 'train', 'validation' and 'test' subsets:
<br>

![Data input folder structure](./assets/images/folder_structure_for_dataset.jpg)

* Average image size in the 'train' set:
<br>

![Averagee image size in the 'train' set](./assets/images/avg_image_size_in_train.png)

* Average and variability images of 'bleached' corals (example - 'dead' and 'healthy' are quite similar visually and not shown):
<br>

![Average and variability of 'bleached' corals](./assets/images/avg_variab_bleached.png)

* Difference between average 'healthy' and 'bleached' images ('healthy'-'dead' and 'dead'-'bleached' look similar and are not shown here):
<br>

![Difference between average 'healthy' and 'bleached'](./assets/images/avg_healthy-bleached_difference.png)

* Montage of randomly selected images of healthy corals (example, 'bleached' and 'dead' are not whown here):
<br>

![Random montage of 'healthy' images](./assets/images/montage_healthy.png)

* Number of images of different labels per set:
<br>

![Images per set](./assets/images/number_in_sets.png)

* Augmented training image (example):
<br>

![Augmented training image](./assets/images/augmented_validation.png)

* For model assessment, building, evaluation and prediction, see [ML Pipeline](#ii-ml-pipeline-1)


## Model optimisation

### Model 1
The training results indicate that the model is performing well, with loss dropping from 0.65 to 0.3 and accuracy increasing from 0.75 to 0.9. The fact that training stopped early (after 12 epochs) suggests that early stopping was used effectively to prevent overfitting, as indicated by the matching train and validation graphs.
#### Analysis of Current Performance
* **Loss Reduction:** A decrease in loss from 0.65 to 0.3 is a substantial improvement.
* **Accuracy Improvement:** An accuracy increase from 0.75 to 0.9 is quite significant.
* **Early Stopping:** The model stopped training early, indicating that it reached a point where further training did not yield significant improvements on the validation set.
* **No Overfitting/Underfitting:** The alignment of training and validation graphs suggests that the model generalizes well to unseen data.

### Model 2



### Model 3


## User Interface design

## Deployment

## References
1. [S. Jamil, M. Rahman, A. Haider, <em>Bag of features (BoF) based deep learning framework for bleached corals detection</em>, Big Data Cogn. Comput. <b>2021</b>, 5, 53.](https://www.mdpi.com/2504-2289/5/4/53)
