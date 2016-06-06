from scipy.misc import imread, imresize
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, TimeDistributedDense, Merge, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils import np_utils

import os
import cnns

max_caption_len = 16

print 'Compiling VGGnet model...'

vggnet = cnns.vggnet_16('keras')

vggnet.load_weights('vgg16_weights.h5')

vggnet.layers.pop()
vggnet.layers.pop()

vggnet.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print 'Model compiled!'

image_file_to_image = {}
image_file_to_features = {}
image_file_to_captions = {}

print 'Indexing and extracting features for image files...'

text_filepath = 'Flickr8k_text/Flickr8k.token.txt'
i = 0
for line in open(text_filepath):
	i += 1
	if i % 100 == 0:
		print 'Processing image %d...' % (i,)
	img_filename, tokens = line.split('\t')
	img_filename = img_filename.split('#')[0]
	if img_filename[-1] != 'g':
		continue
	tokens = tokens.strip()
	tokens = tokens.split(' ')
	caption = ' '.join(tokens[:-1]).lower()
	if img_filename in image_file_to_image:
		image_file_to_captions[img_filename].append(caption)
	else:
		# im = imread('Flicker8k_Dataset/' + img_filename)
		# if len(im.shape) != 3:
		# 	continue
		# im = imresize(im, (244, 244))
		# im = im.transpose((2, 0, 1))
		# im = np.expand_dims(im, axis=0)
		# im = im.astype('float32')
		# # im /= 255
		# features = vggnet.predict(im)
		# image_file_to_image[img_filename] = im
		# image_file_to_features[img_filename] = features
		image_file_to_captions[img_filename] = [caption]

print 'Done indexing and extracting feaures!'

print 'Assembling training set...'

word_dict = {'UNUSED': 0, 'UNKNOWN': 1}
next_available_word_idx = 2
image_features_train = None
partial_captions_train = None
curr_batch_image_features = None
curr_batch_partial_captions = None
next_words_train = []
i = 0
for line in open('Flickr8k_text/Flickr_8k.trainImages.txt'):
	i += 1
	if i % 100 == 0:
		print 'Adding image %d to training set...' % (i,)
		if image_features_train is None:
			image_features_train = curr_batch_image_features
			partial_captions_train = curr_batch_partial_captions
		else:
			image_features_train = np.concatenate([image_features_train, curr_batch_image_features], axis=0)
			partial_captions_train = np.concatenate([partial_captions_train, curr_batch_partial_captions], axis=0)
		curr_batch_image_features = None
		curr_batch_partial_captions = None
	img_filename = line.strip()
	if img_filename[-1] != 'g':
		continue
	im = imread('Flicker8k_Dataset/' + img_filename)
	if len(im.shape) != 3:
		continue
	im = imresize(im, (244, 244))
	im = im.transpose((2, 0, 1))
	im = np.expand_dims(im, axis=0)
	im = im.astype('float32')
	# im /= 255
	features = vggnet.predict(im)
	# features = image_file_to_features[img_filename]
	captions = image_file_to_captions[img_filename]
	for caption in captions:
		tokens = caption.split()
		tokens = ['BEGIN'] + tokens + ['END']
		while len(tokens) < max_caption_len:
			tokens.append('UNUSED')
		for partial_idx in xrange(1, max_caption_len):
			partial_caption = tokens[:partial_idx]
			curr_partial_caption = np.zeros((1, max_caption_len))
			for j in xrange(len(partial_caption)):
				word = tokens[j]
				if word in word_dict:
					curr_partial_caption[0, j] = word_dict[word]
				else:
					word_dict[word] = next_available_word_idx
					next_available_word_idx += 1
					curr_partial_caption[0, j] = word_dict[word]
			next_word = tokens[partial_idx]
			curr_next_word = None
			if next_word in word_dict:
				curr_next_word = word_dict[next_word]
			else:
				word_dict[next_word] = next_available_word_idx
				next_available_word_idx += 1
				curr_next_word = word_dict[next_word]

			if curr_batch_image_features is None:
				curr_batch_image_features = features
				curr_batch_partial_captions = curr_partial_caption
			else:
				curr_batch_image_features = np.concatenate([curr_batch_image_features, features], axis=0)
				curr_batch_partial_captions = np.concatenate([curr_batch_partial_captions, curr_partial_caption], axis=0)

			next_words_train.append(curr_next_word)

if curr_batch_image_features is not None:
	image_features_train = np.concatenate([image_features_train, curr_batch_image_features], axis=0)
	partial_captions_train = np.concatenate([partial_captions_train, curr_batch_partial_captions], axis=0)

vocab_size = len(word_dict)
next_words_train = np_utils.to_categorical(next_words_train, vocab_size)

reverse_word_dict = {}
for word in word_dict:
	word_num = word_dict[word]
	reverse_word_dict[word_num] = word

print 'Done assembling training set!'

print 'Building model...'

image_model = Sequential()
image_model.add(Dense(128, input_shape=(4096,)))
image_model.add(Activation('relu'))

language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
language_model.add(GRU(output_dim=128, return_sequences=True))
language_model.add(TimeDistributedDense(128))


image_model.add(RepeatVector(max_caption_len))


model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
model.add(GRU(256, return_sequences=False))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd)

print 'Done building model!'

print 'Training model...'

model.fit([image_features_train, partial_captions_train], next_words_train, batch_size=16, nb_epoch=100) # orig 100

print 'Done training model!'

# print 'Saving model...'

# model.save_weights('my_model_weights.h5')

# print 'Done saving model!'

print 'Evaluating model...'

predicted_captions = []
reference_captions = []
i = 0
for line in open('Flickr8k_text/Flickr_8k.devImages.txt'):
	i += 1
	if i % 100 == 0:
		print 'Evaluating image %d...' % (i,)
	img_filename = line.strip()
	if img_filename[-1] != 'g':
		continue
	im = imread('Flicker8k_Dataset/' + img_filename)
	if len(im.shape) != 3:
		continue
	im = imresize(im, (244, 244))
	im = im.transpose((2, 0, 1))
	im = np.expand_dims(im, axis=0)
	im = im.astype('float32')
	# im /= 255
	features = vggnet.predict(im)
	# features = image_file_to_features[img_filename]
	captions = image_file_to_captions[img_filename]
	# if len(captions) < 5:
	# 	continue
	predicted_partial_captions = np.zeros((1, max_caption_len))
	predicted_partial_captions[0, 0] = word_dict['BEGIN']
	predicted_caption_tokens = []
	j = 1
	while j < max_caption_len:
		probs = model.predict([features, predicted_partial_captions])
		next_word_num = np.argmax(probs)
		next_word = reverse_word_dict[next_word_num]
		if next_word == 'END':
			break
		predicted_partial_captions[0, j] = next_word_num
		predicted_caption_tokens.append(next_word)
		j += 1

	predicted_caption = ' '.join(predicted_caption_tokens)
	predicted_captions.append(predicted_caption)
	reference_captions.append(captions)
	print 'Caption for image %s: %s' % (img_filename, predicted_caption)

# (Credit to Andrej Karpathy for the BLEU evaluation code below)
open('eval/output', 'w+').write('\n'.join(predicted_captions))
for q in xrange(1):
	open('eval/reference'+`q`, 'w+').write('\n'.join([x[q] for x in reference_captions]))
# invoke the perl script to get BLEU scores
print 'invoking eval/multi-bleu.perl script...'
owd = os.getcwd()
os.chdir('eval')
os.system('./multi-bleu.perl reference < output')
os.chdir(owd)
