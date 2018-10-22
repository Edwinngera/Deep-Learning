from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

# build a model which have 3 convolution layers
model = Sequential()

model.add(InputLayer(input_shape=[64,64,1]))
model.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=50,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=80,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
optimizer = Adam(lr=1e-4)

## for the following model, you can try it if you are interested

# model = Sequential()
# model.add(ZeroPadding2D((1, 1),
#                         input_shape=(64,
#                         64,1)))
#
# model.add(Convolution2D(64, 3, 3, activation='relu',
#                         name='conv1_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(64, 3, 3, activation='relu',
#                         name='conv1_2'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, 3, 3, activation='relu',
#                         name='conv2_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, 3, 3, activation='relu',
#                         name='conv2_2'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu',
#                         name='conv3_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu',
#                         name='conv3_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu',
#                         name='conv3_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu',
#                         name='conv4_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu',
#                         name='conv4_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu',
#                         name='conv4_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu',
#                         name='conv5_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu',
#                         name='conv5_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu',
#                         name='conv5_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Flatten())
#
# # Plugging new Layers
# model.add(Dense(768, activation='sigmoid'))
# model.add(Dropout(0.0))
# model.add(Dense(768, activation='sigmoid'))
# model.add(Dropout(0.0))
# model.add(Dense(2,activation='sigmoid'))
# optimizer = Adam(lr=1e-4, epsilon=1e-08)





