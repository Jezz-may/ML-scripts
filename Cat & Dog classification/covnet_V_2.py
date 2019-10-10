'''

script that changes the number of layers, the number of nodes per layer

we could also change optimizer, with the optimizer we could change the learning rate,
dense layers- whether they are included or not, activation units, kernal size, stride

'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time

X_train = np.load('catdog_features.npy')  # loading X_train
y_train = np.load('catdog_labels.npy')  # loading y_train

X_train = X_train/255  # maximum value a pixel can be is 255, so scaling it by this.

dense_layers = [0, 1]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            name = '{}-conv-{}-nodes-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs\\{}'.format(name))
            # starting to build the model
            model = Sequential()
            model.add(Conv2D(layer_size, (3, 3), input_shape=X_train.shape[1:]))  # first layer is a convolutional layer. here we have a 3x3 window. skipping the first -1 from how the data was reshaped before pickling
            model.add(Activation('relu'))  # rectified linear unit activation function
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))  # rectified linear unit activation function
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten()) # flattening features from 2d to 1d for the dense layer
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.15, callbacks=[tensorboard])
