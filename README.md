# CORALS HEALTH

![Coral Reef](./assets/images/coral-reef-title-image.jpg)
[Image source](https://www.anses.fr/en/content/coral-reefs-french-overseas-territories)

#### INTRODUCTION

Welcome to the "Coral's Health" ML project. This project was inspired by several visits of the author to the Great Barrier Reef (Australia) and observations of detrimental effect (a.k.a. bleaching) of human activity on various colonies of corals. The aim of the project:<br>
* Using an available pre-labelled dataset (suitable for supervised ML) of images of 'Healthy', 'Bleached' and 'Dead' coral colonies, build and train ML model (or models) for the automated prediction of the state of health of unseen coral(s).

This goal will be achieved via addressing two business objectives:
1. Compare average images of different labels ('Healthy', 'Bleached', 'Dead', in pairs) (data visualisation task), and
2. Develop and train ML model which would predict the group where the picture unseen by a model belongs (ML categorisation task).
<hr>



## Table of Contents

### [I. CRISP-DM Methodology](#crisp-dm-methodology)
* [1. Business Understanding](#business-understanding)
* [2. Data Understanding](#data-understanding)
* [3. Data Preparation](#data-preparation)
* [4. Modelling](#modelling)
* [5. Evaluation](#evaluation)
* [6. Deployment](#deployment)

### [II. ML Pipeline](#ii-ml-pipeline-1)
* [1. Data Collection](#data-collection)
* [2. Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
* [3. Feature Engineering](#feature-engineering)
* [4. Model Building](#model-building)
* [5. Model Evaluation](#model-evaluation)


### [III. Jupyter Notebooks](#iii-jupyter-notebooks)
* [1. Notebook Structure](#notebook-structure)
* [2. Results and Visualisation](#results-and-visualisation)




### [IV. Streamlit Dashboard](#iv-streamlit-dashboard-1)
* [1. Overview](#overview)
* [2. Installation and Setup](#installation-and-setup)
* [3. Backend Code: Features and Validation](#backend-code-features-and-validation)
* [4. Dashboard Features]()
* [5. User Guide](#user-guide)

### [V. Conclusions and Future Work](#conclusions-and-future-work)

### [VI. Dependencies and Requirements](#dependencies-and-requirements)

### [VII. How to Run the Project](#how-to-run-the-project)

### [VIII. Deployment](#deployment-2)

#### Acknowledgements

#### Contact information

#### [References](#references-1)

#### [Notes](#notes)
<hr>
<hr>
### CRISP-DM Methodology

#### Business Understanding
Coral bleaching is the process when corals become white due to loss of symbiotic algae and photosynthetic pigments. This loss of pigment can be caused by various stressors, such as changes in ocean temperature (due to Global Warming), light, or nutrients. Bleaching occurs when coral polyps expel the zooxanthellae (dinoflagellates that are commonly referred to as algae) that live inside their tissue, causing the coral to turn white. The zooxanthellae are photosynthetic, and as the water temperature rises, they begin to produce reactive oxygen species. This is toxic to the coral, so the coral expels the zooxanthellae. Since the zooxanthellae produce the majority of coral colouration, the coral tissue becomes transparent, revealing the coral skeleton made of calcium carbonate. Most bleached corals appear bright white, but some are blue, yellow, or pink due to pigment proteins in the coral. According to the United Nations Environment Programme, between 2014 and 2016, the longest recorded global bleaching events killed coral on an unprecedented scale. In 2016, bleaching of coral on the Great Barrier Reef killed 29 to 50 percent of the reef's coral. In 2017, the bleaching extended into the central region of the reef. The average interval between bleaching events has halved between 1980 and 2016, [Wikipedia article](https://en.wikipedia.org/wiki/Coral_bleaching)."
<br>
<br>
The project has two business requirements:<br>
<hr>
<b>1. - The client is interested in having a capability to compare average images obtained for 'healthy', 'bleached' and 'dead' corals and check if these groups can be visually unambiguously categorised.</b>
<hr>
<b>2. - Answer (by using a trained ML model) if an uploaded (previously unseen) image was taken for a 'Healthy', 'Bleached' or 'Dead' coral.</b>
<hr>

* Rationale to map the business requirements to the Data Visualisations and ML tasks:<br>

Addressing these requirements sucessfully will (a)reduce an amount of tedious/repetitive work, (b) reduce involvement of an operator/technician and therefore minimize a human error, and (c) will help with the task to a visually-impaired user. For instance, color-blindeness can make a task of manual sorting of images quite challenging.


#### Data Understanding
Several datasets from [Kaggle](https://www.kaggle.com) were reviewed, primarily containing images labeled as 'Healthy' and 'Bleached'. However, to address the three-category classification ('Healthy', 'Bleached', 'Dead'), the [BHD Corals dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals) with 1,572 labeled images was selected. Initial analysis indicated potential labeling inconsistencies, necessitating further data preparation. It was observed in particular that 'Bleached' corals group partially encompasses two subgroups: 'Bleached' and 'Dead'. The aim of the project and the idea of the author was to check if a ML model (or models) can be extended and trained to enable gategorical classification of images of corals into these **three** groups ('Bleached', 'Healthy' and 'Dead') rather than just **two** ('Healthy' and 'Bleached').<br>
The datasets where coral images would be pre-labelled according to those three categories are much more scarce. This dataset was selected therefore despite some cross contamination which author was expecting to accommodate. The images in the selected dataset were preprocessed by the owner of dataset to have identical 227 x 227 dimension. Data preparation included: (i) cleaning (ii) augmentation.

#### Data Preparation
Data preparation involved cleaning the dataset by removing non-image files and grayscale images, followed by manual verification to correct mislabeled entries. Due to the limited number of images, data augmentation techniques were applied to enhance the dataset's diversity and improve model generalisation.

#### Modelling
Convolutional Neural Networks (CNNs) were employed to classify coral images into three categories. Three models of increasing complexity were developed to mitigate overfitting and enhance generalization. Techniques such as dropout layers and data augmentation were utilized to improve model performance.

#### Evaluation
Models were evaluated based on accuracy and loss metrics using validation and test datasets. The evaluation aimed to determine the models' ability to generalize to unseen data and their effectiveness in correctly classifying coral health states.

#### Deployment
The trained models were deployed using a Streamlit dashboard, enabling users to upload coral images and receive real-time health predictions. This deployment facilitates easy access and usability for stakeholders.



### II. ML Pipeline

#### Data Collection
Complete code used for data collection can be found in [1_data_collection.ipynb](./jupyter_notebooks/1_data_collection.ipynb). Downloaded and unzipped dataset was then checked and processed as follows:
* Cleaning:
1) Check if the files which are not images are present (and deleting them).
2) Checking if the dataset contains grayscale images (and deleting them).
3) [Manual data check](#-1-).
* Dataset splitting:
After data cleaning, the dataset was split into 'train', 'validadion' and 'test' subsets in the ratio 0.7/0.1/0.2. (see [1_data_collection.ipynb](./jupyter_notebooks/1_data_collection.ipynb)).

#### Exploratory Data Analysis
The dataset does not contain any additional information, such as geographic, climate or environmental data, which could be used as an additional assistance in the classification task. In the [project by Jamil <em>et al.</em>](https://www.mdpi.com/2504-2289/5/4/53), the authors too rely solely on pre-labelled image dataset.

#### Feature Engineering
Standard preprocessing steps were applied, including image resizing and normalisation. No additional feature engineering was performed beyond data augmentation to enhance dataset diversity. The [dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals) found for this project contains 1572 labelled images. See corresponding section in [Jupyter notebooks, resulst and visualisation](#results-and-visualisation) chapter for visual representation of image number distribution in various subsets and labels. The total number is very modest which is the cause of poor generalisation initially observed for trained model. In order to improve model training and generalisation when analysing previously unseen data, augmentation process was implemented [2_visualisation.ipynb](./jupyter_notebooks/2_visualisation.ipynb).

#### Model Building
Three CNN models were developed with increasing complexity to address overfitting:

- **Model_1**: Basic CNN inspired by the [Malaria Detector](https://github.com/Code-Institute-Solutions/WalkthroughProject01) project.
- **Model_2**: Added dropout layers to reduce overfitting.
- **Model_3**: Further increased model complexity with additional convolutional layers and dropout.

Refer to the [Models Evaluation Notebook](./jupyter_notebooks/3_models_evaluation_cross-compare.ipynb) for detailed architectures and training processes.

The ML task (categorising (more than two classes or labels)) set in this project will have to be addressed using the CNN-type model. At this point it is iportant to provide a high-level visualisation of data life cycle workflow. Up to now, the data were follofing a single pathway. At the modelling step, three models (of increased complexity) were built and trained. The data life cycle is shown chematically below:

![data life cycle workflow](./assets/images/data_cycle_workflow.jpg)

**Model_1** Was inspired by the the model used in [MalariaDetector walkthrough project](https://github.com/Code-Institute-Solutions/WalkthroughProject01.git) as a starting point. In this model, binary_crossentropy loss was substituted by categorical_crossentropy.
```
def create_tf_model_1():
    """
    Function that will create, compile and return a sequential model for classifying
    three types of images of corals ('healthy', 'bleached' and 'dead').
    """
    model = Sequential()    

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))        

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))            

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))            

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
```
Despite seemingly reasonable training, the model demonstrated poor generalisation when shown unseen data. The author speculates that this is due to the model was originally developed for binary classification and more ample supply of training data. Poor generalising might be, the most likely the result of overfitting.

To tackle this problem: (i) The data [augmentation was used](#-augmentation), (ii) dropout layers were added (**Model_2**), and (iii) another, yet more complex model with extra deep layer (combination of Conv2D, MaxPooling2D and Dropout) was built (**Model_3**). In particular, adding Dropout layers is supposed to suppress overfitting. These two latter models are shown below:

**Model_2**
```
def create_tf_model_2():
    """
    Function that will create, compile and return a sequential model for classifying
    three types of images of corals ('healthy', 'bleached' and 'dead'). Model_2 has additional
    Dropout() layers compared to Model_1.
    """
    model = Sequential()    

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))           
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))        

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))        

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
```
**Model_3**
```
def create_tf_model_3():
    """
    Function that will create, compile and return a sequential model for classifying
    three types of images of corals ('healthy', 'bleached' and 'dead'). The model
    has an extra set of layers:
    ----------
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    ----------
    compared to the Model2.
    """
    model = Sequential()    

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))      

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))        

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))        

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
```

#### Model Evaluation
Model_1, Model_2 and Model_3 were evaluated using the test set and accuracy and loss metrics. The results are shown below:

**Model_1**
```
loss: 0.4456 - accuracy: 0.8249

```
**Model_2**
```
loss: 0.4423 - accuracy: 0.8081

```
**Model_3**
```
loss: 0.5171 - accuracy: 0.8519

```
As expected, **model_3** gave slightly better accuracy-loss balance.

The loss and accuracy curves for loss and accuracy during training and validation are given below:

**Model_1:**

![Model_1_accuracy](./assets/images/model_1_training_acc.png) ![Model_1_loss](./assets/images/model_1_training_losses.png)

**Model_2:**

![Model_2_accuracy](./assets/images/model_2_training_acc.png) ![Model_2_loss](./assets/images/model_2_training_losses.png)

**Model_3:**

![Model_3_accuracy](./assets/images/model_3_training_acc.png) ![Model_3_loss](./assets/images/model_3_training_losses.png)

### III. Jupyter Notebooks

#### Notebook Structure
The project is described and summarised in three notebooks:<br>
1. Data Collection
2. Visualisation
3. Models evaluation and cross comparing.

<u><em>Data collection</em></u> notebook contain the code for:

* Setting up folder structure of the project,  acquiring the dataset from [Kaggle.com](https://www.kaggle.com/datasets/sonainjamil/bhd-corals)
* Two-step automated data cleaning: checking for presence of files which are not images and checking if images are not grayscale (this would result in code breaking during further data analysis)
* Splitting the dataset into training, validation and test subsets in ratio 0.7/0.1/0.2 ratio, respectively.

<u><em>Visualisation</em></u> notebook answers the 1st business requirement: <b>"The client is interested in having a capability to compare average images obtained for 'healthy', 'bleached' and 'dead' corals and check if these groups can be visually unambiguously categorised</b>". It contains the code for:
* Visualisation of image data (average image size across the entire dataset)
* Functions to load images, shapes and labels in NumPy array
* Plotting and saving mean and variability images for each label
* Plot difference images between average images in pairs 'dead'-'bleached', 'dead'-'healthy' and 'bleached'-'healthy'
* Plot montage of random images for 'healthy', 'dead' and 'bleached' labels.

<u><em>Models evaluation and cross comparing</em></u> notebook answers the 2nd business requirement: <b>"Can we tell if an uploaded (previously unseen) image was taken for a 'Healthy', 'Bleached' or 'Dead' coral?"</b>. It contains the code for:
* Plotting the distribution of image numbers in train, validation and test data subsets for different labels in the dataset
* Generating and plotting augmentation data
* Building the models (including defining the early stop criteria)
* Models training
* Models performance assessment
* Models evaluation
* Prediction using the new data and plotting probability prediction for different models.

#### Results and Visualisation
This paragraphs summarises graphical outcomes of the jupyter notebooks described above.

* Dataset folder structure after splitting the dataset into the 'train', 'validation' and 'test' subsets:
<br>

![Data input folder structure](./assets/images/folder_structure_for_dataset.jpg)

* Average image size in the 'train' set:
<br>

![Averagee image size in the 'train' set](./assets/images/avg_image_size_in_train.png)

* Average and variability images of 'bleached' corals (example - 'dead' and 'healthy' are visually identical):
<br>

![Average and variability of 'bleached' corals](./assets/images/avg_variab_bleached.png)

* Difference between average 'healthy' and 'bleached' images ('healthy'-'dead' and 'dead'-'bleached' pairs look identical):
<br>

![Difference between average 'healthy' and 'bleached'](./assets/images/avg_healthy-bleached_difference.png)

* Montage of randomly selected images of healthy (as an example) corals:
<br>

![Random montage of 'healthy' images](./assets/images/montage_healthy.png)

* Number of images of different labels per set:
<br>

![Images per set](./assets/images/number_in_sets.png)

* Augmented training image (example):
<br>

![Augmented training image](./assets/images/augmented_validation.png)

* For model assessment, building, evaluation and prediction, see [ML Pipeline](#ii-ml-pipeline-1)

### IV. Streamlit Dashboard

#### Overview
Streamlit dashboard is a multipage app featuring the following pages/sections:
1. Business Case
2. Project Hypothesis
3. Corals Visualiser
4. Corals State Identifier
5. ML Performance Metrics
6. Models Summary

The user can navigate between sections using the dropdown menu:

![Navigation dropdown list](./assets/images/streamlit-dropdown-list.jpg)

#### Installation and Setup
The app is deployed [here](https://corals-health-3265febe1ec7.herokuapp.com/), is ready to use and does not require additional setup or configuring steps.

#### Backend Code: Features and Validation
The backend code can be found in the [project repo](https://github.com/DrSYakovlev/corals_health.git).
The app pages are located in the [app_pages](https://github.com/DrSYakovlev/corals_health/tree/main/app_pages) and [src](https://github.com/DrSYakovlev/corals_health/tree/main/src) folders.
The code was validated using [CI Python Linter](https://pep8ci.herokuapp.com/) with no errors of warnings flagged.

#### Dashboard features

#### User Guide
1. Using the dropdown menu, navigate to the 'Business Case', 'Project Hypothesis', 'ML Performance Metrics' or 'Model Summary' pages for more details, business requirements, model performance and structure. These sections are mostly informative.
2. Navigate to the 'Corals Visualiser' to view typical/representative images of corals from different labelled groups.

-> Check 'Difference between averages for different groups' box to visualise the difference between averages in pairs 'Healthy' - 'Bleached', 'Healthy' - 'Dead' and 'Bleached' - 'Dead'.

-> Check 'Image Montage' box, select label from the dropdown list anc click 'Create Montage' to generate randon collection of images of corresponding group.

3. Navigate to the 'Corals State Identifier' to run prediction on previously unseen/unused images of corals. Simply upload or drag and drop image or images where indicated. It may take 10-15 seconds to process data, depending on you interned speed.

### Conclusions and Future Work





### Dependencies and Requirements

For a successful deploymetnt on [Heroku](https://www.heroku.com/) , the next files are needed:

[setup.sh](https://github.com/DrSYakovlev/corals_health/blob/main/setup.sh)

[Procfile](https://github.com/DrSYakovlev/corals_health/blob/main/Procfile)

[runtime.txt](https://github.com/DrSYakovlev/corals_health/blob/main/runtime.txt)

[requirements.txt](https://github.com/DrSYakovlev/corals_health/blob/main/requirements.txt)


### How to Run the Project

1. **Clone the Repository**
    ```bash
    git clone https://github.com/DrSYakovlev/corals_health.git
    cd corals_health
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Jupyter Notebooks**
    - Open and run the notebooks in the `jupyter_notebooks` directory. Run this step if you want to build and (re)train your own model. Please, be advised that the dataset was a also a subject of manual sorting ([see this chapter](#data-collection)). The models [1](https://github.com/DrSYakovlev/corals_health/blob/main/outputs/v4/model_1.h5), [2](https://github.com/DrSYakovlev/corals_health/blob/main/outputs/v4/model_2.h5), and [3](https://github.com/DrSYakovlev/corals_health/blob/main/outputs/v4/model_3.h5) are deployed in the project repo and accessed in the deployed app via their urls due to the excessive size.
4. **Launch the Streamlit Dashboard**
    ```bash
    streamlit run app.py
    ```

5. **Access the Dashboard**
    - Open the provided URL in your browser to interact with the dashboard.


### Deployment

This project was deployed in a manual mode following the steps, described [here](https://medium.com/featurepreneur/how-to-connect-github-to-heroku-be6ff27419d3). The deployed app can be found here: [Corals Health](https://corals-health-3265febe1ec7.herokuapp.com/).

#### Acknowledgements




#### Contact information

[LinkedIn](https://www.linkedin.com/in/sergey-yakovlev-823514295/)


## References
1. [S. Jamil, M. Rahman, A. Haider, <em>Bag of features (BoF) based deep learning framework for bleached corals detection</em>, Big Data Cogn. Comput. <b>2021</b>, 5, 53.](https://www.mdpi.com/2504-2289/5/4/53)

### Comments

### -1-

During the initial attempts of the model_1 (see [Model Building section](#model-building)) training and validation, it was discovered that the **Model_1** behaves well and stops training according to pre-defined stopping criteria. However, the model showed quite pure generalisation when processing unseen data. This could be an indication of overfitting. This dictated the need to have a closer look at the dataset. During manual check, it was discovered that the [dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals) contains some images in wrong folders.  The dataset used in this work was assembled and preprocessed in the context of the project, published by [Jamil <em>et al.</em>](https://www.mdpi.com/2504-2289/5/4/53). Although, the [dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals) of coral images is labelled as 'Healthy', 'Bleached' and 'Dead', the work was focused on distinguishing beween 'Healthy' and 'Bleached' (binary classification task) using 'specific deep convolutional neural networks such as AlexNet, GoogLeNet, VGG-19, ResNet-50, Inception v3, and CoralNet. (c)' The subset labelled as 'Dead' was treated as 'Bleached'. Attempt to train the model to categorise the data into three groups: 'Bleached', 'Dead' and 'Healthy' (this work) resulted in poor genaralisation and overfitting. Manual inspection of the dataset revealed that some of the 'Dead' corals were labelled as 'Bleached' and the other way around. Futhermore, some of the 'Bleached' corals were marked as 'Healthy'. This image misplacement may be less critical for binary classification, but crucial for training models for more categories. Therefore, the author had to visually inspect the entire dataset and manually move some images in downloaded dataset where the misplacement was obvous, into more appropriate folders, following the [description](https://en.wikipedia.org/wiki/Coral_bleaching). It must also be mentioned that apart from the described dataset cross-contamination, the original dataset contains stranger data, such as this garden stuff: ![garden stuff.](./assets/images/salad_leaf_coral.jpg)








