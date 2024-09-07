import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
# import tempfile
import io
import matplotlib.pyplot as plt
from PIL import Image



def page_models_explained_body():
    st.success("### Summary and structure of models used in the Project")
    version = 'v4'
    model_1 = load_model(f"outputs/{version}/model_1.h5")
    model_2 = load_model(f"outputs/{version}/model_2.h5")
    model_3 = load_model(f"outputs/{version}/model_3.h5")    
    
    st.info("#### Model_1")
    
    stringlist_1 = []
    model_1.summary(print_fn=lambda x: stringlist_1.append(x))
    summary_string_1 = "\n".join(stringlist_1)
    st.text(summary_string_1)
    
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
    
    if st.checkbox('Show Model_1 code'):
        st.code(model_1_code, language='python')
    
        
    st.write("---")
    
    st.info("#### Model_2")
    
    stringlist_2 = []
    model_2.summary(print_fn=lambda x: stringlist_2.append(x))
    summary_string_2 = "\n".join(stringlist_2)
    st.text(summary_string_2)
    
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
    
    if st.checkbox('Show Model_2_code'):
        st.code(model_2_code, language='python')    
    
    st.write("---")
    
    st.info("#### Model_3")
    
    stringlist_3 = []
    model_3.summary(print_fn=lambda x: stringlist_3.append(x))
    summary_string_3 = "\n".join(stringlist_3)
    st.text(summary_string_3)
    
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
    
    if st.checkbox('Show Model_3 code'):
        st.code(model_3_code, language='python')