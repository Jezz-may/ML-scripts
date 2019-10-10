from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time

name = 'catdog-cnn-64x2-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs\\{}'.format(name))

X_train = np.load('catdog_features.npy')  # loading X_train
y_train = np.load('catdog_labels.npy')  # loading y_train

X_train = X_train/255  # maximum value a pixel can be is 255, so scaling it by this.

# starting to build the model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))  # first layer is a convolutional layer. here we have a 3x3 window. skipping the first -1 from how the data was reshaped before pickling
model.add(Activation('relu'))  # rectified linear unit activation function
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))  # second layer is a convolutional layer with a 3x3 window.
model.add(Activation('relu'))  # rectified linear unit activation function
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # flattening features from 2d to 1d for the dense layer
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.15, callbacks=[tensorboard])

'''
20313/20313 [==============================] - 458s 23ms/sample - loss: 0.6303 - accuracy: 0.6312 - val_loss: 0.5305 - val_accuracy: 0.7423
20313/20313 [==============================] - 427s 21ms/sample - loss: 0.4169 - accuracy: 0.8107 - val_loss: 0.3614 - val_accuracy: 0.8343
20313/20313 [==============================] - 446s 22ms/sample - loss: 0.1808 - accuracy: 0.9315 - val_loss: 0.1401 - val_accuracy: 0.9481
'''
