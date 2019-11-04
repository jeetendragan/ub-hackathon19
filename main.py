import sys

# sys.path.append('/usr/local/lib/python2.7/dist-packages/')

# print "\n".join(sys.path)

import cv2 as cv2
import numpy as np # linear algebra
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pickle
import datetime
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.models import Sequential, Input
from keras import layers
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D,Conv2D

from keras.utils import to_categorical


from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
# from sklearn.metrics import log_loss
from numpy.random import permutation

import keras
import keras.backend as K 
from keras.models import Model
from keras.layers import Dense, Dropout, Add, Input, BatchNormalization, Activation
from keras.layers import  Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.regularizers import l2

import cv2
import matplotlib.pyplot as plt
import glob

import os

def vgg_std16_model(img_rows, img_cols, color_type=3):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols,color_type)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    #model.load_weights('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main_block(x, filters, n, strides, dropout):
    # Normal part
    x_res = Conv2D(filters, (3,3), strides=strides, padding="same")(x)# , kernel_regularizer=l2(5e-4)
    x_res = BatchNormalization()(x_res)
    x_res = Activation('relu')(x_res)
    x_res = Conv2D(filters, (3,3), padding="same")(x_res)
    # Alternative branch
    x = Conv2D(filters, (1,1), strides=strides)(x)
    # Merge Branches
    x = Add()([x_res, x])

    for i in range(n-1):
        # Residual conection
        x_res = BatchNormalization()(x)
        x_res = Activation('relu')(x_res)
        x_res = Conv2D(filters, (3,3), padding="same")(x_res)
        # Apply dropout if given
        if dropout: x_res = Dropout(dropout)(x)
        # Second part
        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)
        x_res = Conv2D(filters, (3,3), padding="same")(x_res)
        # Merge branches
        x = Add()([x, x_res])

    # Inter block part
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_model(input_dims, output_dim, n, k, act= "relu", dropout=None):
    """ Builds the model. Params:
            - n: number of layers. WRNs are of the form WRN-N-K
                 It must satisfy that (N-4)%6 = 0
            - k: Widening factor. WRNs are of the form WRN-N-K
                 It must satisfy that K%2 = 0
            - input_dims: input dimensions for the model
            - output_dim: output dimensions for the model
            - dropout: dropout rate - default=0 (not recomended >0.3)
            - act: activation function - default=relu. Build your custom
                   one with keras.backend (ex: swish, e-swish)
    """
    # Ensure n & k are correct
    assert (n-4)%6 == 0
    assert k%2 == 0
    n = (n-4)//6 
    # This returns a tensor input to the model
    inputs = Input(shape=(input_dims))

    # Head of the model
    x = Conv2D(16, (3,3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3 Blocks (normal-residual)
    x = main_block(x, 16*k, n, (1,1), dropout) # 0
    x = main_block(x, 32*k, n, (2,2), dropout) # 1
    x = main_block(x, 64*k, n, (2,2), dropout) # 2
            
    # Final part of the model
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    outputs = Dense(output_dim, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_image(path, rows=None, cols=None, gray=True):
    if gray:
        img = cv2.imread(path,0)
    else:
        img = cv2.imread(path)
    if rows != None and cols != None:
        img = cv2.resize(img,(rows,cols))
        #img = np.reshape(img, (rows, cols,1))
    return img

model = build_model((224,224,3), 10,16,4)
model.load_weights('weights.h5')
model.compile("adam","categorical_crossentropy", ['accuracy'])

def load_data(split=0.33, rows=None, cols=None):
    paths = glob.glob(os.path.join("../input/state-farm-distracted-driver-detection", "train", "",  ".jpg"))
    labels = [int(x.split('/')[4][1]) for x in paths]
    if rows != None and cols != None:
        images = [load_image(x, rows, cols,gray=False) for x in paths]
    else:
        images = [load_image(x, gray=False) for x in paths]
    y = to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=split)
    
    return np.array(x_train), np.array(x_test), y_train, y_test

img_rows = 224
img_cols = 224
# load_data(rows=img_rows, cols=img_cols)



cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 1
defaultPos = [10, 10]
while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    # frame = cv2.resize(frame,(224,224))

    # out_mapping = ['c0', 'c1', 'c2', 'c3', ]
    c = cv2.waitKey(1)
    if c == 99:
        #draw_label(frame, "Captured", defaultPos)
        img = cv2.resize(frame, (224, 224))
        out = model.predict(np.array([np.array(img)]))

        print(out)
        cv2.putText(frame,str(np.argmax(out)), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.putText(img,str(np.argmax(out)), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        cv2.imwrite("frame%d.jpg" % count, img)
        count += count

    cv2.imshow('Input', frame)
    if c == 27:
        break

cap.release()       
cv2.destroyAllWindows()