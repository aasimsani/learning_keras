# A small experiment with the Happy house dataset
An experiment to learn Keras on a bored evening when I wanted to understand how I could work on ML faster


Here's the Keras summary for the rudimentary CNN for the Happy house project

```Train Accuracy: 100%```
```Test Accuracy: 96.67%```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        544
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 16)        2064
_________________________________________________________________
P1 (MaxPooling2D)            (None, 7, 7, 16)          0
_________________________________________________________________
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
dropout (Dropout)            (None, 784)               0
_________________________________________________________________
D1 (Dense)                   (None, 256)               200960
_________________________________________________________________
D2 (Dense)                   (None, 2)                 514
=================================================================
Total params: 204,082
Trainable params: 204,082
Non-trainable params: 0
_________________________________________________________________
````
## How to train?

1) Clone this repository
2) Run ``` pip3 install -r requirements.txt```
3) Make a folder called ```dataset/``` in the root of this repository

### If you want to retrain with the same dataset
4) Setup the kaggle API https://github.com/Kaggle/kaggle-api
5a) If the kaggle API is setup and works run ```download_data.sh```
5b) If it doen't work download the dataset from here: https://www.kaggle.com/iarunava/happy-house-dataset 
6) Once the ```dataset/``` folder has the Happy house dataset files you can run ```model.py```

### If you want to use your own dataset
4) Parse your images into (64,64,3) numpy arrays
5) Create a hdf5 file of the dataset. Split it into train and test files and place them in the datasets folder
6) You can adjust the dataset paths in the ```helper.py``` file's ```data()``` function
7) Once all of this is done just run ```model.py```

## How to predict?
1) Just run ```predict.py``` and it'll access your webcame via OpenCV

NOTE: If you have your own model and want to use it for prediction. It'll be stored in the ```models/```

## Extras

I highly recommend using [Comet ML](https://www.comet.ml/) to train and iterate. It saves a lot of time.

You can create a file called ```config.py``` and place the API key for your project in a variable named ```COMET_API_KEY``` and the model.py file will automatically use it.

