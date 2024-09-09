import requests
import os
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

def download_model(url, local_path):
    """
    Helper function to download a model file from a URL and save it locally.
    """
    # Ensure the target folder exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Get the model file
    response = requests.get(url, allow_redirects=True)
    with open(local_path, 'wb') as file:
        file.write(response.content)
    print("Downloaded model to:", local_path)

def page_models_explained_body():
    st.success("### Summary and structure of models used in the Project")
    version = 'v4'

    model_1_url = 'https://github.com/DrSYakovlev/corals_health/raw/main/outputs/v4/model_1.h5'
    model_2_url = 'https://github.com/DrSYakovlev/corals_health/raw/main/outputs/v4/model_2.h5'
    model_3_url = 'https://github.com/DrSYakovlev/corals_health/raw/main/outputs/v4/model_3.h5'

    # Local paths where the models will be saved
    model_1_path = 'saved_models/model_1.h5'
    model_2_path = 'saved_models/model_2.h5'
    model_3_path = 'saved_models/model_3.h5'

    # Download models if they don't exist locally
    if not os.path.exists(model_1_path):
        download_model(model_1_url, model_1_path)
    if not os.path.exists(model_2_path):
        download_model(model_2_url, model_2_path)
    if not os.path.exists(model_3_path):
        download_model(model_3_url, model_3_path)

    # Load the models
    model_1 = load_model(model_1_path)
    model_2 = load_model(model_2_path)
    model_3 = load_model(model_3_path)

    # Function to summarize model architecture
    def summarize_model(model, model_name):
        st.info(f"#### {model_name}")
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        summary_string = "\n".join(stringlist)
        st.text(summary_string)

    # Summarize each model
    
    model_1_code = '''
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
    '''
    
    model_2_code = '''
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
    '''
    
    model_3_code = '''
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
    '''
    
    summarize_model(model_1, "Model_1")
    if st.checkbox('Show Model_1 code'):
        st.code(model_1_code, language='python')        
    st.write("---")
    summarize_model(model_2, "Model_2")
    if st.checkbox('Show Model_2_code'):
        st.code(model_2_code, language='python')
    st.write("---")
    summarize_model(model_3, "Model_3")        
    if st.checkbox('Show Model_3 code'):
        st.code(model_3_code, language='python')
