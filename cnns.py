# This code was written by myself, with the Lasagne model zoo as reference
# on certain nets (namely, GoogLeNet and VGG16).
# For models with pretrained weights given by Lasagne,
# the layer names match those given by the Lasagne model zoo.

# Caution: not all of these (especially the large ones) have been tested
# thoroughly.

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import FlattenLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax, linear
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer

def single_layer(library, img_channels, img_rows, img_cols, nb_classes):
	if library == 'keras':
		model = Sequential()

		model.add(Flatten(input_shape=(img_channels, img_rows, img_cols)))
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((img_channels, img_rows, img_cols))

		net['fc'] = DenseLayer(net['input'], num_units=nb_classes)
		net['prob'] = NonlinearityLayer(net['fc'], softmax)

		return net


def mlp(library, img_channels, img_rows, img_cols, nb_classes):
	if library == 'keras':
		model = Sequential()

		# First fully connected layer
		model.add(Flatten(input_shape=(img_channels, img_rows, img_cols)))
		model.add(Dense(512))
		model.add(Activation('relu'))

		# Softmax prediction layer
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((img_channels, img_rows, img_cols))

		# First fully connected layer
		net['fc1'] = DenseLayer(net['input'], num_units=512)

		# Softmax prediction layer
		net['fc2'] = DenseLayer(net['fc1'], num_units=nb_classes, nonlinearity=None)
		net['prob'] = NonlinearityLayer(net['fc2'], softmax)

		return net


def lenet(library, img_channels, img_rows, img_cols, nb_classes):
	nb_filters = 32
	nb_pool = 2
	nb_conv = 3

	if library == 'keras':
		model = Sequential()

		# First convolutional layer
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
														border_mode='valid',
														input_shape=(img_channels, img_rows, img_cols)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

		# Second convolutional layer
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

		# Fully connected layer
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))

		# Softmax prediction layer
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((img_channels, img_rows, img_cols))

		# First convolutional layer
		net['conv1'] = ConvLayer(
				net['input'], nb_filters, nb_conv, pad=1, flip_filters=False)
		net['pool1'] = PoolLayer(net['conv1'], nb_pool)

		# Second convolutional layer
		net['conv2'] = ConvLayer(
				net['pool1'], nb_filters, nb_conv, pad=1, flip_filters=False)
		net['pool2'] = PoolLayer(net['conv2'], nb_pool)

		# Third convolutional layer
		net['fc1'] = DenseLayer(net['pool2'], num_units=512)


		net['fc2'] = DenseLayer(net['fc1'], num_units=nb_classes, nonlinearity=None)
		net['prob'] = NonlinearityLayer(net['fc2'], softmax)

		return net


# Some other simple LeNet-like CNNs I tried

def mycnn1(library, img_channels, img_rows, img_cols, nb_classes):
	nb_filters = 32
	nb_pool = 2
	nb_conv = 3

	if library == 'keras':
		model = Sequential()

		# First convolutional layer
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
		                        border_mode='valid',
		                        input_shape=(img_channels, img_rows, img_cols)))
		model.add(Activation('relu'))

		# Second convolutional layer
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
		model.add(Dropout(0.25))

		# Fully connected layer
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		# Softmax prediction layer
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((img_channels, img_rows, img_cols))

		# First convolutional layer
		net['conv1'] = ConvLayer(
				net['input'], nb_filters, nb_conv, flip_filters=False)

		# Second convolutional layer
		net['conv2'] = ConvLayer(
				net['conv1'], nb_filters, nb_conv, flip_filters=False)
		net['pool2'] = PoolLayer(net['conv2'], nb_pool)
		net['pool2_dropout'] = DropoutLayer(net['pool2'], p=0.25)

		# Fully connected layer
		net['fc1'] = DenseLayer(net['pool2_dropout'], num_units=128)
		net['fc1_dropout'] = DropoutLayer(net['fc1'], p=0.5)

		# Softmax prediction layer
		net['fc2'] = DenseLayer(
				net['fc1_dropout'], num_units=nb_classes, nonlinearity=None)
		net['prob'] = NonlinearityLayer(net['fc2'], softmax)

		return net


def mycnn2(library, img_channels, img_rows, img_cols, nb_classes):
	if library == 'keras':
		model = Sequential()

		# First convolutional layer
		model.add(Convolution2D(32, 3, 3, border_mode='same',
		                        input_shape=(img_channels, img_rows, img_cols)))
		model.add(Activation('relu'))

		# Second convolutional layer
		model.add(Convolution2D(32, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Third convolutional layer
		model.add(Convolution2D(64, 3, 3, border_mode='same'))
		model.add(Activation('relu'))

		# Fourth convolutional layer
		model.add(Convolution2D(64, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Fully connected layer
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		# Softmax prediction layer
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((img_channels, img_rows, img_cols))

		# First convolutional layer
		net['conv1'] = ConvLayer(net['input'], 32, 3, flip_filters=False)

		# Second convolutional layer
		net['conv2'] = ConvLayer(net['conv1'], 32, 3, flip_filters=False)
		net['pool2'] = PoolLayer(net['conv2'], 2)
		net['pool2_dropout'] = DropoutLayer(net['pool2'], p=0.25)

		# Third convolutional layer
		net['conv3'] = ConvLayer(net['pool2_dropout'], 64, 3, flip_filters=False)

		# Fourth convolutional layer
		net['conv4'] = ConvLayer(net['conv3'], 64, 3, flip_filters=False)
		net['pool4'] = PoolLayer(net['conv4'], 2)
		net['pool4_dropout'] = DropoutLayer(net['pool4'], p=0.25)

		# Fully connected layer
		net['fc1'] = DenseLayer(net['pool4_dropout'], num_units=512)
		net['fc1_dropout'] = DropoutLayer(net['fc1'], p=0.5)

		# Softmax prediction layer
		net['fc2'] = DenseLayer(
				net['fc1_dropout'], num_units=nb_classes, nonlinearity=None)
		net['prob'] = NonlinearityLayer(net['fc2'], softmax)

		return net


def mycnn3(library, img_channels, img_rows, img_cols, nb_classes):
	if library == 'keras':
		model = Sequential()

		# First convolutional layer
		model.add(Convolution2D(16, 5, 5,
														input_shape=(img_channels, img_rows, img_cols)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# Second convolutional layer
		model.add(Convolution2D(16, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# Third convolutional layer
		model.add(Convolution2D(32, 3, 3))
		model.add(Activation('relu'))

		# Softmax prediction layer
		model.add(Flatten())
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((img_channels, img_rows, img_cols))

		# First convolutional layer
		net['conv1'] = ConvLayer(net['input'], 16, 5, flip_filters=False)
		net['pool1'] = PoolLayer(net['conv1'], 2)

		# Second convolutional layer
		net['conv2'] = ConvLayer(net['pool1'], 16, 3, flip_filters=False)
		net['pool2'] = PoolLayer(net['conv2'], 2)

		# Third convolutional layer
		net['conv3'] = ConvLayer(net['pool2'], 32, 3, flip_filters=False)

		# Softmax prediction layer
		net['fc1'] = DenseLayer(
				net['conv3'], num_units=nb_classes, nonlinearity=None)
		net['prob'] = NonlinearityLayer(net['fc1'], softmax)

		return net


def mycnn4(library, img_channels, img_rows, img_cols, nb_classes):
	if library == 'keras':
		model = Sequential()

		# First convolutional layer
		model.add(Convolution2D(32, 5, 5,
														input_shape=(img_channels, img_rows, img_cols)))
		model.add(Activation('relu'))

		# Second convolutional layer
		model.add(Convolution2D(32, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# Third convolutional layer
		model.add(Convolution2D(64, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# Fourth convolutional layer
		model.add(Convolution2D(128, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# Fully connected layer
		model.add(Flatten())
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		# Softmax prediction layer
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((img_channels, img_rows, img_cols))

		# First convolutional layer
		net['conv1'] = ConvLayer(net['input'], 32, 5, flip_filters=False)

		# Second convolutional layer
		net['conv2'] = ConvLayer(net['conv1'], 32, 3, flip_filters=False)
		net['pool2'] = PoolLayer(net['conv2'], 2)

		# Third convolutional layer
		net['conv3'] = ConvLayer(net['pool2'], 64, 3, flip_filters=False)
		net['pool3'] = PoolLayer(net['conv3'], 2)

		# Fourth convolutional layer
		net['conv4'] = ConvLayer(net['pool3'], 128, 3, flip_filters=False)
		net['pool4'] = PoolLayer(net['conv4'], 2)

		# Fully connected layer
		net['fc1'] = DenseLayer(net['pool4'], num_units=256)
		net['fc1_dropout'] = DropoutLayer(net['fc1'], p=0.5)

		# Softmax prediction layer
		net['fc2'] = DenseLayer(
				net['fc1_dropout'], num_units=nb_classes, nonlinearity=None)
		net['prob'] = NonlinearityLayer(net['fc2'], softmax)

		return net


def mycnn5(library, img_channels, img_rows, img_cols, nb_classes):
	if library == 'keras':
		model = Sequential()

		# First convolutional layer
		model.add(Convolution2D(32, 5, 5,
														input_shape=(img_channels, img_rows, img_cols)))
		model.add(Activation('relu'))

		# Second convolutional layer + max pool
		model.add(Convolution2D(64, 5, 5))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# Third convolutional layer
		model.add(Convolution2D(64, 3, 3))
		model.add(Activation('relu'))

		# Fourth convolutional layer
		model.add(Convolution2D(64, 3, 3))
		model.add(Activation('relu'))

		# Fifth convolutional layer + max pool
		model.add(Convolution2D(64, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# Sixth convolutional layer
		model.add(Convolution2D(128, 3, 3))
		model.add(Activation('relu'))

		# First fully connected layer
		model.add(Flatten())
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		# Second fully connected layer
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		# Softmax prediction layer
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((img_channels, img_rows, img_cols))

		# First convolutional layer
		net['conv1'] = ConvLayer(net['input'], 32, 5, flip_filters=False)

		# Second convolutional layer + max pool
		net['conv2'] = ConvLayer(net['conv1'], 64, 5, flip_filters=False)
		net['pool2'] = PoolLayer(net['conv2'], 2)

		# Third convolutional layer
		net['conv3'] = ConvLayer(net['pool2'], 64, 3, flip_filters=False)

		# Fourth convolutional layer
		net['conv4'] = ConvLayer(net['conv3'], 64, 3, flip_filters=False)

		# Fifth convolutional layer + max pool
		net['conv5'] = ConvLayer(net['conv4'], 64, 3, flip_filters=False)
		net['pool5'] = PoolLayer(net['conv5'], 2)

		# Sixth convolutional layer
		net['conv6'] = ConvLayer(net['pool5'], 128, 3, flip_filters=False)

		# First fully connected layer
		net['fc1'] = DenseLayer(net['conv6'], num_units=256)
		net['fc1_dropout'] = DropoutLayer(net['fc1'], p=0.5)

		# Second fully connected layer
		net['fc2'] = DenseLayer(net['fc1_dropout'], num_units=256)
		net['fc2_dropout'] = DropoutLayer(net['fc2'], p=0.5)

		# Softmax prediction layer
		net['fc3'] = DenseLayer(
				net['fc2_dropout'], num_units=nb_classes, nonlinearity=None)
		net['prob'] = NonlinearityLayer(net['fc3'], softmax)

		return net



## The following models are specific to ImageNet - images must be 244 x 244


def alexnet(library):
	if library == 'keras':
		model = Sequential()

		# First convolutional layer
		model.add(Convolution2D(96, 11, 11, border_mode='valid', subsample=(4, 4),
														input_shape=(3, 224, 224)))
		model.add(Activation('relu'))
		model.add(BatchNormalization(alpha=0.000002, k=1))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

		# Second convolutional layer
		model.add(Convolution2D(256, 5, 5, border_mode='valid'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(alpha=0.000002, k=1))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

		# Third convolutional layer
		model.add(Convolution2D(384, 3, 3, border_mode='valid'))
		model.add(Activation('relu'))

		# Fourth convolutional layer
		model.add(Convolution2D(384, 3, 3, border_mode='valid'))
		model.add(Activation('relu'))

		# Fifth convolutional layer
		model.add(Convolution2D(256, 3, 3, border_mode='valid'))
		model.add(Activation('relu'))

		# First fully connected layer
		model.add(Flatten())
		model.add(Dense(4096))
		model.add(Activation('relu'))

		# Second fully connected layer
		model.add(Dense(4096))
		model.add(Activation('relu'))

		# Softmax prediction layer
		model.add(Dense(1000))
		model.add(Activation('softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((3, 244, 244))

		# First convolutional layer
		net['conv1'] = ConvLayer(
				net['input'], 96, 11, stride=(4, 4), flip_filters=False)
		net['norm1'] = LRNLayer(net['conv1'])
		net['pool1'] = PoolLayer(net['norm1'], 3, stride=2)

		# Second convolutional layer
		net['conv2'] = ConvLayer(net['pool1'], 256, 5, flip_filters=False)
		net['norm2'] = LRNLayer(net['conv2'])
		net['pool2'] = PoolLayer(net['norm2'], 3, stride=2)

		# Third convolutional layer
		net['conv3'] = ConvLayer(net['pool2'], 384, 3, flip_filters=False)

		# Fourth convolutional layer
		net['conv4'] = ConvLayer(net['conv3'], 384, 3, flip_filters=False)

		# Fifth convolutional layer
		net['conv5'] = ConvLayer(net['conv4'], 256, 3, flip_filters=False)

		# First fully connected layer
		net['fc1'] = DenseLayer(net['conv5'], num_units=4096)

		# Second fully connected layer
		net['fc2'] = DenseLayer(net['fc1'], num_units=4096)

		# Softmax prediction layer
		net['fc3'] = DenseLayer(net['fc2'], num_units=1000, nonlinearity=None)
		net['prob'] = NonlinearityLayer(net['fc3'], softmax)

		return net


# This model can be initialized with pre-trained weights in the
# Lasagne model zoo
def googlenet(library):
	if library == 'keras':
		model = Sequential()

		# First convolutional layer
		model.add(ZeroPadding2D(padding=(3, 3), input_shape=(3, 244, 244)))
		model.add(Convolution2D(64, 7, 7, subsample=(2, 2)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(BatchNormalization(alpha=0.000002, k=1))

		# Second convolutional layer
		model.add(Convolution2D(64, 1, 11))
		model.add(Activation('relu'))
		model.add(ZeroPadding2D(padding=(1, 1)))
		model.add(Convolution2D(192, 3, 3))
		model.add(Activation('relu'))
		model.add(BatchNormalization(alpha=0.000002, k=1))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

		# First set of inception modules
		create_inception_module_keras(model, [32, 64, 96, 128, 16, 32])
		create_inception_module_keras(model, [64, 128, 128, 192, 32, 96])
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

		# Second set of inception modules
		create_inception_module_keras(model, [64, 192, 96, 208, 16, 48])
		create_inception_module_keras(model, [64, 160, 112, 224, 24, 64])
		create_inception_module_keras(model, [64, 128, 128, 256, 24, 64])
		create_inception_module_keras(model, [64, 122, 144, 288, 32, 64])
		create_inception_module_keras(model, [128, 256, 160, 320, 32, 128])
		model.add(pool_size=(3, 3), strides=(2, 2))

		# Third set of inception modules
		create_inception_module_keras(model, [128, 256, 160, 320, 32, 128])
		create_inception_module_keras(model, [128, 384, 192, 384, 48, 128])

		# Average pooling layer
		model.add(AveragePooling2D())

		# Softmax prediction layer
		model.add(Flatten())
		model.add(Dense(1000))
		model.add(Activation('softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((None, 3, 244, 244))

		# First convolutional layer
		net['conv1/7x7_s2'] = ConvLayer(
				net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
		net['pool1/3x3_s2'] = PoolLayer(
				net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
		net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.000002, k=1)

		# Second convolutional layer
		net['conv2/3x3_reduce'] = ConvLayer(
				net['pool1/norm1'], 64, 1, flip_filters=False)
		net['conv2/3x3'] = ConvLayer(
				net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
		net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
		net['pool2/3x3_s2'] = PoolLayer(
				net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

		# First set of inception modules
		net.update(create_inception_module_lasagne('inception_3a',
																							 net['pool2/3x3_s2'],
																							 [32, 64, 96, 128, 16, 32]))
		net.update(create_inception_module_lasagne('inception_3b',
																							 net['inception_3a/output'],
																							 [64, 128, 128, 192, 32, 96]))
		net['pool3/3x3_s2'] = PoolLayer(
				net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)

		# Second set of inception modules
		net.update(create_inception_module_lasagne('inception_4a',
																							 net['pool3/3x3_s2'],
																							 [64, 192, 96, 208, 16, 48]))
		net.update(create_inception_module_lasagne('inception_4b',
																							 net['inception_4a/output'],
																							 [64, 160, 112, 224, 24, 64]))
		net.update(create_inception_module_lasagne('inception_4c',
																							 net['inception_4b/output'],
																							 [64, 128, 128, 256, 24, 64]))
		net.update(create_inception_module_lasagne('inception_4d',
																							 net['inception_4c/output'],
																							 [64, 122, 144, 288, 32, 64]))
		net.update(create_inception_module_lasagne('inception_4e',
																							 net['inception_4d/output'],
																							 [128, 256, 160, 320, 32, 128]))
		net['pool4/3x3_s2'] = PoolLayer(
				net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)

		# Third set of inception modules
		net.update(create_inception_module_lasagne('inception_5a',
																							 net['pool4/3x3_s2'],
																							 [128, 256, 160, 320, 32, 128]))
		net.update(create_inception_module_lasagne('inception_5b',
																							 net['inception_5a/output'],
																							 [128, 384, 192, 384, 48, 128]))

		net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
		net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'], num_units=1000,
																				 nonlinearity=None)
		net['prob'] = NonlinearityLayer(net['loss3/classifier'],
																		nonlinearity=softmax)

		return net


def create_inception_module_keras(model, num_filters):
	pool_model.add(ZeroPadding2D(padding=(1, 1)))
	pool_model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
	pool_model.add(Convolution2D(num_filters[0], 1, 1))
	pool_model.add(Activation('relu'))

	# 1x1
	one_by_one_model.add(Convolution2D(num_filters[1], 1, 1))
	one_by_one_model.add(Activation('relu'))

	# 3x3
	three_by_three_model.add(Convolution2D(num_filters[2], 1, 1))
	three_by_three_model.add(Activation('relu'))
	three_by_three_model.add(ZeroPadding2D(padding=(1, 1)))
	three_by_three_model.add(Convolution2D(num_filters[3], 3, 3))
	three_by_three_model.add(Activation('relu'))

	# 5x5
	five_by_five_model.add(Convolution2D(num_filters[4], 1, 1))
	five_by_five_model.add(Activation('relu'))
	five_by_five_model.add(ZeroPadding2D(padding=(2, 2)))
	five_by_five_model.add(Convolution2D(num_filters[5], 5, 5))
	five_by_five_model.add(Activation('relu'))

	model.add(Merge[pool_model, one_by_one_model,
									three_by_three_model, five_by_five_model])


def create_inception_module_lasagne(name, prev_name, num_filters):
	pass
	# net = {}
	# net['pool'] = PoolLayer(prev_name, pool_size=3, pad=1, stride=1)
	# net['pool_proj'] = ConvLayer(
	# 	net['pool'], num_filters[0], 1, flip_filters=False)

	# # 1x1
	# net['1x1'] = ConvLayer(prev_name, num_filters[1], 1, flip_filters=False)

	# # 3x3
	# net['3x3_reduce'] = ConvLayer(
	# 		prev_name, num_filters[3], 3, pad=1, flip_filters=False)
	# net['3x3'] = ConvLayer(
	# 	net['3x3_reduce'], num_filters[3], 3, pad=1, flip_filters=False)

	# # 5x5
	# net['5x5_reduce'] = ConvLayer(
	# 		input_layer, num_filters[4], 1, flip_filters=False)
	# net['5x5'] = ConvLayer(
	# 		net['5x5_reduce'], num_filters[5], 5, pad=2, flip_filters=False)

	# net['output'] = ConcatLayer([
	# 		net['1x1'],
	# 		net['3x3'],
	# 		net['5x5'],
	# 		net['pool_proj']
	# ])

 #  return {'%s/%s' % (name, layer_name) : net[layer_name] for layer_name in net}


# This network can be weight-inialized using downloaded weights
# in either Keras or Lasagne
def vggnet_16(library):
	if library == 'keras':
		model = Sequential()

		model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
		model.add(Convolution2D(64, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(64, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(128, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(128, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(256, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(256, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(256, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, 3, 3, activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(Flatten())
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1000, activation='softmax'))

		return model
	else:
		net = {}
		net['input'] = InputLayer((None, 3, 224, 224))

		net['conv1_1'] = ConvLayer(
				net['input'], 64, 3, pad=1, flip_filters=False)
		net['conv1_2'] = ConvLayer(
				net['conv1_1'], 64, 3, pad=1, flip_filters=False)
		net['pool1'] = PoolLayer(net['conv1_2'], 2)

		net['conv2_1'] = ConvLayer(
				net['pool1'], 128, 3, pad=1, flip_filters=False)
		net['conv2_2'] = ConvLayer(
				net['conv2_1'], 128, 3, pad=1, flip_filters=False)
		net['pool2'] = PoolLayer(net['conv2_2'], 2)

		net['conv3_1'] = ConvLayer(
				net['pool2'], 256, 3, pad=1, flip_filters=False)
		net['conv3_2'] = ConvLayer(
				net['conv3_1'], 256, 3, pad=1, flip_filters=False)
		net['conv3_3'] = ConvLayer(
				net['conv3_2'], 256, 3, pad=1, flip_filters=False)
		net['pool3'] = PoolLayer(net['conv3_3'], 2)

		net['conv4_1'] = ConvLayer(
				net['pool3'], 512, 3, pad=1, flip_filters=False)
		net['conv4_2'] = ConvLayer(
				net['conv4_1'], 512, 3, pad=1, flip_filters=False)
		net['conv4_3'] = ConvLayer(
				net['conv4_2'], 512, 3, pad=1, flip_filters=False)
		net['pool4'] = PoolLayer(net['conv4_3'], 2)

		net['conv5_1'] = ConvLayer(
				net['pool4'], 512, 3, pad=1, flip_filters=False)
		net['conv5_2'] = ConvLayer(
				net['conv5_1'], 512, 3, pad=1, flip_filters=False)
		net['conv5_3'] = ConvLayer(
				net['conv5_2'], 512, 3, pad=1, flip_filters=False)
		net['pool5'] = PoolLayer(net['conv5_3'], 2)

		net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
		net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)

		net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
		net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)

		net['fc8'] = DenseLayer(
				net['fc7_dropout'], num_units=1000, nonlinearity=None)
		net['prob'] = NonlinearityLayer(net['fc8'], softmax)

		return net

