from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

import cloudpickle
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

# force tensorflow to work on CPU as GPU doesn't work for now
tf.config.set_visible_devices([], 'GPU')

batch_size = 256
epochs = 4
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets

# TODO : use mnist.load_data to create x_train, y_train, x_test, y_test
(x_train, y_train), (x_test, y_test) = mnist.load_data()


if K.image_data_format() == 'channels_first':
    x_train = np.expand_dims(x_train, axis = 1)
    x_test = np.expand_dims(x_test, axis = 1)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = np.expand_dims(x_train, axis = 3)
    x_test = np.expand_dims(x_test, axis = 3)
    input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

y_train = tf.keras.utils.to_categorical(
    y_train, num_classes=10
)

y_test = tf.keras.utils.to_categorical(
    y_test, num_classes=10
)

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
                 ])

model.compile(optimizer = "Adam",
              loss = "categorical_crossentropy" ,
              metrics = ["accuracy"]
                 )

model.fit(x_train, y_train ,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test ))

score = model.evaluate(x_test, y_test)



import mlflow
import mlflow.sklearn
import mlflow.tensorflow
mlflow.end_run()

with mlflow.start_run():

    print("Accuracy: {}".format(score))
    
    mlflow.log_metric("Accuracy", score)

    # Log model 
    mlflow.tensorflow.log_model(model, "model")