import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical


def rgb2gray(rgb):
    """
    Function used to convert a RGB image to Grayscale
    :param rgb: A numpy array of the RGB image
    :returns: A numpy array with a grayscale image
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def data():
    """
    Function to load dataset 
    :returns: Training and testing data
    """
    path_to_train = "./dataset/happy-house-dataset/train_happy.h5"
    path_to_test = "./dataset/happy-house-dataset/test_happy.h5"

    # Load training data
    train_dataset = h5py.File(path_to_train)
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    # Load testing data
    test_dataset = h5py.File(path_to_test)
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    # Convert to Black and White
    train_x_bw = []
    test_x_bw = []

    for image in train_x:
        train_x_bw.append(rgb2gray(image))

    for image in test_x:
        test_x_bw.append(rgb2gray(image))

    # Appropriately reshape the array to be digested 
    train_x_bw = np.reshape(np.array(train_x_bw),(600,64,64,1))
    test_x_bw = np.reshape(np.array(test_x_bw),(150,64,64,1))

    # Turn to one hot categorical vectors
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)


    return train_x_bw, train_y, test_x_bw, test_y



