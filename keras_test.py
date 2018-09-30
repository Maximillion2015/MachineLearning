from keras import *
# from keras.datasets import cifar10
from keras.layers import Dense
from keras.layers import Activation
import keras

model = Sequential()
model.add(Dense(input_dim=28*28, output_dim=500))
model.add(Activation('sigmoid'))

model.add(Dense(output_dim=500))
model.add(Activation('sigmoid'))

model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model.fit(x_train, y_train, batch_size=100, nb_epoch=20)

score = model.evaluate(x_test, y_test)
print 'Total loss on Testing Set:', score[0]
print 'Accuracy Testing Set:', score[1]
