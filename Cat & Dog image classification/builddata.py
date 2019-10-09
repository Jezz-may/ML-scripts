'''
Download the following data set from an old competition on Kaggle
https://www.kaggle.com/c/dogs-vs-cats/data
This script prepares the data so that it is in a workable state and ready for a model to be applied to it.
'''

import numpy as np
import matplotlib.pyplot as plt  # used to show the images
import os  # used for directories and paths
import cv2  # used for image operations
from tqdm import tqdm
import random
import pickle

directory = 'C:/Users/jerom/OneDrive/Documents/machinelearning/sentdextut/TensorFlow/classifydogcat/data sets/PetImages'
labels = ['Dog', 'Cat']

# creating training data
training_data = []  # creating training data as an empty array


def build_training_data(image_size):
    '''
    function to create the training data. The images will be converted so that they are square and in greyscale.
    :param image_size: the x dimension that the image will be resized to (x by x image)
    '''
    for label in labels:  # iterating through each label
        path = os.path.join(directory, label)  # path to the dog or cat directory
        label_number = labels.index(label)  # converting the labels to numbers (dog=0,cat=1)
        for image in tqdm(os.listdir(path)):  # iterate through each image
            try:
                image_arr = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)  # converting image into black and white  ## note the american spelling
                square_image_arr = cv2.resize(image_arr, (image_size, image_size))  # resizing image to be square
                training_data.append([square_image_arr, label_number])  # adding image and it's label to training data
            except Exception as e:  # for simplicity reasons
                pass


image_size = 128
build_training_data(image_size)

random.shuffle(training_data)  # shuffling data so that it's all mixed up
#for image in training_data[-10:]: print(image[1])  # short list of labels showing that they're all mixed up.

# making the feature and label data set
X_train = []
y_train = []
for feature, label in training_data:
    X_train.append(feature)
    y_train.append(label)
X_train = np.array(X_train).reshape(-1, image_size, image_size, 1)  # for Keras it needs a numpy array and not a list

# saving the data so this script doesn't have to be run every time.
np.save('catdog_features.npy', X_train)  # saving X_train
np.save('catdog_labels.npy', y_train)  # saving y_train

'''
X_train = np.load('catdog_features.npy')  # loading X_train
y_train = np.load('catdog_labels.npy')  # loading y_train

#alternative, can do via pickling
pickle_out = open('X_train.pickle','wb')
pickle.dump(X_train, pickle_out)
pickle_out.close()
pickle_out = open('y_train.pickle','wb')
pickle.dump(y_train, pickle_out)
pickle_out.close()
# code to load into script
pickle_in = open('catdog_X.pickle','rb')
X_train = pickle.load(pickle_in)
pickle_in = open('y_train.pickle','rb')
y_train = pickle.load(pickle_in)
'''