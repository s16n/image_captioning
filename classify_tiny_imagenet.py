import os
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from scipy.misc import imread, imresize
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import cnns

class_names = []
class_name_to_class_num = {}
f = open('wnids.txt')
i = 0
for line in f:
	class_name = line.strip()
	class_names.append(class_name)
	class_name_to_class_num[class_name] = i
	i += 1

X_train = None
y_train = []
X_test = None
y_test = []

for class_num, class_name in enumerate(class_names):
	print 'Getting images for class:', class_name, class_num
	base_dir = os.getcwd() + '/train/' + class_name + '/images/'
	i = 0
	class_images = None
	for filename in os.listdir(base_dir):
		# if i >= 100:
		# 	break
		im = imread(base_dir + filename)
		if len(im.shape) != 3:
			continue
		im = im.transpose((2, 0, 1))
		im = np.expand_dims(im, axis=0)
		if class_images is None:
			class_images = im
		else:
			class_images = np.concatenate([class_images, im], axis=0)
		y_train.append(class_num)
		i += 1
	if X_train is None:
		X_train = class_images
	else:
		X_train = np.concatenate([X_train, class_images], axis=0)

f = open(os.getcwd() + '/val/val_annotations.txt')
i = 0
this_batch = None
for line in f:
	if i % 1000 == 999:
		print 'Getting val image %d' % (i + 1,)
		if X_test is None:
			X_test = this_batch
		else:
			X_test = np.concatenate([X_test, this_batch], axis=0)
		this_batch = None
	tokens = line.split('\t')
	filename = tokens[0]
	class_name = tokens[1]
	im = imread(os.getcwd() + '/val/images/' + filename)
	if len(im.shape) != 3:
		continue
	im = im.transpose((2, 0, 1))
	im = np.expand_dims(im, axis=0)
	if this_batch is None:
		this_batch = im
	else:
		this_batch = np.concatenate([this_batch, im], axis=0)
	y_test.append(class_name_to_class_num[class_name])
	i += 1
if this_batch is not None:
	X_test = np.concatenate([X_test, this_batch], axis=0)

batch_size = 128
nb_classes = len(class_names)
nb_epoch = 200

data_augmentation = False

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model = cnns.mycnn1('keras', 3, 64, 64, nb_classes)

# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd)

if not data_augmentation:
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              show_accuracy=True,
              verbose=1,
              shuffle=True)
else:

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)

    datagen.fit(X_train)

    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        show_accuracy=True,
                        verbose=1,
                        validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
print(score)
print('Test score:', score[0])
print('Test accuracy:', score[1])
