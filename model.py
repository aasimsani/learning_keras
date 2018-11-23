
from comet_ml import Experiment
import time

import tensorflow as tf 
import numpy as np
from tensorflow.keras import layers,optimizers,regularizers,initializers,models
from tensorflow.keras.utils import to_categorical
import h5py

from helper import rgb2gray,data

# Use comet ML if defined
try: 
    from config import COMET_API_KEY
    experiment = Experiment(api_key=COMET_API_KEY,
                        project_name="general", workspace="aasimsani")

except ImportError as e:
    pass



print(tf.VERSION)
print(tf.keras.__version__)



# Load data
train_x, train_y, test_x, test_y = data()


def create_model(x_train,y_train,x_test,y_test):
    """
    Function to create the ML model
    """

    # Define input size for first layer
    in_size = (64,64,1)

    # Define number of classes predicted
    out_classes = 2


    # Define a sequential model so we can quickly make a model by adding layers to the API
    model = tf.keras.Sequential()

    # Convolve input once
    L1 = layers.Conv2D(filters=32,kernel_size=(4,4),strides=[2,2],input_shape=in_size,activation='relu',
                        kernel_initializer=initializers.glorot_uniform(seed=None),padding='same',
                        kernel_regularizer=regularizers.l2(0.01))
    model.add(L1)

    # Convolve input again
    L2 = layers.Conv2D(filters=16,kernel_size=(2,2),strides=[2,2],input_shape=in_size,activation='relu',
                        kernel_initializer=initializers.glorot_uniform(seed=None),padding='same',
                        kernel_regularizer=regularizers.l2(0.01))
    model.add(L2)


    # Pool the convolutions and extract important parts
    P1 = layers.MaxPooling2D(pool_size=3,strides=2, padding='valid',name="P1")
    model.add(P1)

    # Flatten the pooled layer
    F = layers.Flatten(name="flatten")
    model.add(F)

    # Add dropout to the flattened layers to generalize better
    dO = layers.Dropout(0.01)
    model.add(dO)

    # First dense layer 
    D1 = layers.Dense(256,activation='relu',name='D1',kernel_initializer=initializers.glorot_uniform(seed=None),
                        kernel_regularizer=regularizers.l2(0.01))
    model.add(D1)



    # Output layer
    D2 = layers.Dense(out_classes,activation='softmax', name='D2')
    model.add(D2)

    # Output the structure of the model
    model.summary()


    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizers.Adam(lr=0.00001))


    # 1000 is a lot but since the initialization might different at all times. 1000 guarantees convergence
    result = model.fit(train_x,train_y,epochs=1000,
              verbose=2,validation_data=(test_x,test_y))

    # Store model after training acoording to time it was made
    filepath = "./models/model-"+ str(time.time()) +".h5"
    tf.keras.models.save_model(
        model,
        filepath,
        overwrite=True,
        include_optimizer=True
        )



# Run model training
create_model(train_x,train_y,test_x,test_y)

