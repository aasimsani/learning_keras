
from tensorflow.keras import models
import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

from helper import rgb2gray,data


# Load stored model
path = "./models/model-1542912426.7963462.h5"
new_model = models.load_model(path)
new_model.summary()




train_x, train_y, test_x, test_y = data()


sample1x = np.reshape(test_x[2],(1,64,64,1))
sample1y = test_y[2]

sample2x = np.reshape(test_x[-1],(1,64,64,1))
sample2y = test_y[-1]


print(new_model.predict(sample1x))
print(sample1y)

print(new_model.predict(sample2x))
print(sample2y)

