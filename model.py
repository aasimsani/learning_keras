import tensorflow as tf 
import numpy as np
from tensorflow.keras import layers,optimizers,regularizers,initializers
from tensorflow.keras.utils import to_categorical

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe


import h5py

print(tf.VERSION)
print(tf.keras.__version__)



def data():
    path_to_train = "./dataset/happy-house-dataset/train_happy.h5"
    path_to_test = "./dataset/happy-house-dataset/test_happy.h5"

    train_dataset = h5py.File(path_to_train)
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(path_to_test)
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])


    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    # y reshaped
    # train_y = train_y.reshape((1, train_x.shape[0]))
    # test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y






train_x, train_y, test_x, test_y = data()

print("X shape: ",train_x.shape)
print("Y shape: ", train_y.shape)
# print("Sample X: " ,train_x[0,:,:,:])
# print("Sample Y: ",train_y[:,0:4])



def create_model(x_train,y_train,x_test,y_test):

    in_size = (64,64,3)
    out_classes = 2


    model = tf.keras.Sequential()


    L1 = layers.Conv2D(filters=32,kernel_size=(4,4),strides=[2,2],input_shape=in_size,activation='relu',
                        kernel_initializer=initializers.glorot_uniform(seed=None))
    model.add(L1)

    P1 = layers.MaxPooling2D(pool_size=3,strides=2, padding='same',name="P1")
    model.add(P1)

    L2 = layers.Conv2D(filters=16,kernel_size=(2,2),strides=[2,2],input_shape=in_size,activation='relu',
                        kernel_initializer=initializers.glorot_uniform(seed=None))
    model.add(L2)

    P2 = layers.MaxPooling2D(pool_size=2,strides=2, padding='same',name="P2")

    model.add(P2)

    F = layers.Flatten(name="flatten")
    model.add(F)

    D1 = layers.Dense(256,activation='relu',name='D1',kernel_initializer=initializers.glorot_uniform(seed=None))
    model.add(D1)

    D2 = layers.Dense(out_classes,activation='sigmoid', name='D2')
    model.add(D2)

    model.summary()

    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizers.Adam(lr=0.000003))


    result = model.fit(train_x,train_y,epochs=250,
              verbose=2,validation_data=(test_x,test_y))




create_model(train_x,train_y,test_x,test_y)

