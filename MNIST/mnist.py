'''
This script is a basic TensorFlow model trained and tested on numbers from the mnist data set.
The model has two hidden layers with 256 neurons and a rectified linear unit (ReLU) activation function.
The output layer uses a softmax activation function.
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist  # numbers data set (28 x 28)

(X_train, y_train), (X_test, y_test) = mnist.load_data()  # unpacking data into train and test

#plt.imshow(X_train[0], cmap='binary')  # plot image in black and white to check what it looks like

X_train = tf.keras.utils.normalize(X_train, axis=1)  # normalizing the values in train, range 0 - 1
X_test = tf.keras.utils.normalize(X_test, axis=1)  # normalizing the values in test, range 0 - 1

model = tf.keras.models.Sequential()  # defining a feed forward model
model.add(tf.keras.layers.Flatten())  # change the images from a multidimensional array into a flat array.
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # 256 neurons, rectified linear activation function
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # probability distribution function

# parameters for training the model
model.compile(loss='sparse_categorical_crossentropy',  # using sparse here because targets are integers and not one-hot encoded
              optimizer='adam',  # SGD method based on adaptive estimation of first and second order moments
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=3)

valuation_loss, valuation_accuracy = model.evaluate(X_test, y_test)
print('val_loss:', valuation_loss, 'val_acc: ', valuation_accuracy)  # val_loss: 0.08741461245724931 val_acc:  0.9736

# individual testing for user by eye
pred = model.predict([X_test])
np.argmax(pred[18])
plt.imshow(X_test[18], cmap='binary')
