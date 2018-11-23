
from tensorflow.keras import models
import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

path = "./models/model-1542912426.7963462.h5"
new_model = models.load_model(path)
new_model.summary()



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def data():
    path_to_train = "./dataset/happy-house-dataset/train_happy.h5"
    path_to_test = "./dataset/happy-house-dataset/test_happy.h5"

    train_dataset = h5py.File(path_to_train)
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(path_to_test)
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    train_x_bw = []
    test_x_bw = []

    for image in train_x:
        train_x_bw.append(rgb2gray(image))

    for image in test_x:
        test_x_bw.append(rgb2gray(image))

    
    train_x_bw = np.reshape(np.array(train_x_bw),(600,64,64,1))
    test_x_bw = np.reshape(np.array(test_x_bw),(150,64,64,1))


    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    # y reshaped
    # train_y = train_y.reshape((1, train_x.shape[0]))
    # test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x_bw, train_y, test_x_bw, test_y






train_x, train_y, test_x, test_y = data()


sample1x = np.reshape(test_x[2],(1,64,64,1))
sample1y = test_y[2]

sample2x = np.reshape(test_x[-1],(1,64,64,1))
sample2y = test_y[-1]


print(new_model.predict(sample1x))
print(sample1y)

print(new_model.predict(sample2x))
print(sample2y)

