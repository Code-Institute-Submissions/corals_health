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
