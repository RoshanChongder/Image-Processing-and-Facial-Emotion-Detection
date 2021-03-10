import os
import sys

import pandas as pd
import numpy as np


# keras is free source deep learning library in python
# from this library different layer will be imported
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

# Method to read CSV Files
def read_CSV( path ):
    try:
        file = pd.read_csv( path )
        return file
    except FileNotFoundError:
        print( "CSV File not found at " + path )
        return None
    except Exception:
        print(" Unknown error appeared ")
        return None

# Addition of data into list from CSV for training and public testing
def data_Addition() :
    global x_train , y_train , x_test , y_test , data_set

    for row_count,row in data_set.iterrows():
        value = row['pixels'].split(' ')   # extracting the pixels as a list
        try :
            if 'Training' in row['Usage'] :        # if the current column is for Training
                x_train.append(np.array(value,'float32'))        # adding the pixels in the x axis
                y_train.append(row['emotion'])                        # adding emotion in the y axis
            elif 'PublicTest' in row['Usage']:    # if the current column is for testing
                x_test.append(np.array( value,'float32'))
                y_test.append( row['emotion'])

        except:
            print(" Error occurred at row number " + row_count)
            print("Data Set in that row is " + row )

# Method to Convert from list to Numpy Arrays
def Convert_to_np_Array():
    global x_train, x_test, y_test , y_train
    # Converting list to numpy Array
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')


def Rescale():
    # Normalizing the data
    # why data normalization is required - https://www.import.io/post/what-is-data-normalization-and-why-is-it-important/
    # how it's work - https://www.educative.io/edpresso/data-normalization-in-python
    # read Out - https://www.mathsisfun.com/data/standard-deviation.html

    global x_train , x_test , y_test , y_train

    # we are basically rescaling
    x_train -= np.mean(x_train, axis=0)
    x_train /= np.std(x_train, axis=0)  # CENTRALIZING THE DATA

    x_test -= np.mean(x_test, axis=0)
    x_test /= np.std(x_test, axis=0)


def Reshape( width , height ):

    global x_train , y_train , x_test , y_test

    x_train = x_train.reshape(x_train.shape[0], width, height, 1)
    x_test = x_test.reshape(x_test.shape[0], width, height, 1)
    # WHAT THIS FUNCTION DOES TO_CATEGORICAL
    y_train = np_utils.to_categorical(y_train,num_classes=7)
    y_test = np_utils.to_categorical(y_test,num_classes=7)


def Design_CNN():

    # The number of epochs is a hyperparameter
    # that defines the number times that the learning algorithm will work through the entire training dataset
    features = 64
    Batch_size = 64
    Label = 7
    epoch = 1
    global  x_train, y_train

    model = Sequential()


    ## Layer 1

    # adding layers
    # Conv2d is used as the image are in 2d format

    # here we are trying extract input
    # Relu is a rectifier

    # Search Kernal size
    model.add(Conv2D(features,kernel_size=(3,3),activation='relu',input_shape=(x_train.shape[1:])))
    model.add(Conv2D(features,kernel_size=(3, 3),activation='relu'))

    # adding a max pooling 2D layer
    # It mainly helps to control over fitting
    # can use average pooling layer also
    model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

    # adding a drop out layer
    model.add(Dropout(0.3))

    ## 2ND layer
    model.add(Conv2D(features, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(features, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    ## 3RD Layer
    model.add(Conv2D(2*features, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(2*features, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))



    # What is drop out layer

    model.add( Flatten() )

    # adding dense layers
    model.add(Dense(2**3 * features, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2 ** 3 * features, activation='relu'))
    model.add(Dropout(0.2))

    # Adding the final layers
    model.add(Dense(Label,activation='softmax')) # Activation is softmax as we want to bind in the 7 labels of 0ptions

    model.compile(loss=categorical_crossentropy,optimizer=Adam(),metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=Batch_size,
              epochs=epoch,
              verbose=1,
              validation_data=(x_test,y_test),
              shuffle=True )


    # Saving the model

    EmotionDetectJson = model.to_json()
    with open("fer.json","w") as file :
        file.write(EmotionDetectJson)
    model.save_weights("fer.h5")

























data_set = read_CSV('./DataSet/fer2013/fer2013.csv')

print( data_set.info() )   # checking the data set

print( data_set.head() )   # checking few of the starting rows

'''

Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   emotion  35887 non-null  int64        << To label the EMotion >>  like 0 -> happy 1 -> sad etc 
 1   pixels   35887 non-null  object       << Image - represented in pixels >>   
 2   Usage    35887 non-null  object       << To train or to test >>  
dtypes: int64(1), object(2)
memory usage: 841.2+ KB 


Example On Data - 

   emotion                                             pixels     Usage
0        0  70 80 82 72 58 58 60 63 54 58 60 48 89 115 121...  Training
1        0  151 150 147 155 148 133 111 140 170 174 182 15...  Training
2        2  231 212 156 164 174 138 161 173 182 200 106 38...  Training

'''

# Now we will do training and public testing

x_train , y_train = [] , []   # data the will be used for training will added in this two lists
x_test , y_test = [] , []     # data that will be used for public testing will be added here

data_Addition()    # addition of data in the lists for training and testing

# checking the lists
print( x_train[:2])
print( y_train[:2])

# As the Keras Module only takes numpy arrays as input
# we need to convert this lists into numpy arrays

Convert_to_np_Array()

# Rescalling the data
Rescale()

# Reshaping the x train and y train in to a one d array
Reshape(48,48)

# Start designing Our CNN

# To build the model we will be using sequential Type

Design_CNN()


print("Completed")


# Training 70%
# validation 30 %