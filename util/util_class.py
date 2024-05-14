import re
import os
import numpy as np
import pandas as pd
import string
from string import punctuation

from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gensim import models

import settings




emotions = ['negative', 'positive', 'neutral']
punctuation_list = list(punctuation)
stop_words = list(set(stopwords.words('english')))


def remove_unecesary_data(sent):
	# remove urls (https?:\/\/\S+) --> for urls with http
	sent = re.sub(r'https?:\/\/\S+', '<url>', sent)
	sent = re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '<url>', sent)
	# remove html reference characters
	sent = re.sub(r'&[a-z]+;', '', sent)
	#remove non-letter characters
	#sent = re.sub(r"[a-z\s\(\-:\)\\\/\\];='#", "", sent)
	#removing handles
	sent = re.sub(r'@[a-zA-Z0-9-_]*', '', sent)
	# remove the symbol from hastag to analize the word
	# remove numbers
	sent = re.sub(r'[0-9]+', '', sent)

	sent_aux = ''
	for token in sent.split():
		if token not in stop_words:
			sent_aux += token + ' '
	
	sent = re.sub(r'\s+', ' ', sent.strip())
	
	return sent


def preprocessing_semeval(sent):
	#sent = remove_unecesary_data(sent.lower())

	tknzr = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)
	tokens = tknzr.tokenize(sent)

	#return [w for w in tokens if w not in punctuation_list]
	return tokens


def preprocessing_sst2(sent):
	#global counter_idx, word2idx
	sent = re.sub(r'[' + string.punctuation + ']+', '', sent)
	sent = sent.strip()
	sent = re.sub(r'[^\w\s]', '', sent, re.UNICODE)
	sent = re.sub(r' +', ' ', sent)
	#sent = [w for w in word_tokenize(sent) if w in stop_words]

	return sent


def preprocessing_isear(sent):
	sent = re.sub(r'\n', '', re.sub(r'á', '', sent))
	sent = re.sub(r'\s+', ' ', sent.strip())
	sent = re.sub(r'[' + punctuation + ']+', '', sent)
	sent = re.sub(r'[0-9]+', '', sent)

	sent_aux = ''
	for token in sent.split():
		if token not in stop_words:
			sent_aux += token + ' '

	sent = sent_aux.strip()

	return sent


def read_file(file_name):
	x_val = []
	y_val = []
	for file in os.listdir(file_name):
		#read_csv() expects everything to be wrapped in quotes, quoting=0(default). 
		#If it is a tab separated file with no quotes, the value should be quoting=3
		df = pd.read_csv(file_name + '/' + file, sep='\t', header=None, quoting=3)

		for index, row in df.iterrows():
			x_val.append(str(row[2]))
			y_val.append(str(row[1]))

	x_val = [preprocessing_semeval(sent.lower()) for sent in x_val]

	return x_val, y_val


def read_sem_eval():
	x_train, y_train = read_file(settings.dir_semeval + 'train')
	x_test, y_test = read_file(settings.dir_semeval + 'test')

	dict_class = {}
	for count, value in enumerate(list(set(y_train))):
		dict_class[value] = count

	y_train = np.array([dict_class[val] for val in y_train])
	y_test = np.array([dict_class[val] for val in y_test])

	return x_train, y_train, x_test, y_test, dict_class



def read_sst2():
	datasets = {'train': [], 'dev': [], 'test': []}
	
	for file_ in ['train.csv', 'dev.csv', 'test.csv']:
		key = re.sub('.csv', '', file_)
		df = pd.read_csv(settings.dir_sst2 + file_, sep='|')
		for index, row in df.iterrows():
			datasets[key].append((row[0], str(row[1])))
				
	y_train, x_train = zip(*datasets['train'])
	y_dev, x_dev = zip(*datasets['dev'])
	y_test, x_test = zip(*datasets['test'])

	x_train = [preprocessing_sst2(sent) for sent in x_train]
	x_dev = [preprocessing_sst2(sent) for sent in x_dev]
	x_test = [preprocessing_sst2(sent) for sent in x_test]
	y_train = np.array(y_train)
	y_dev = np.array(y_dev)
	y_test = np.array(y_test)
	

	return x_train, y_train, x_dev, y_dev, x_test, y_test



def read_isear():
	df = pd.read_csv(settings.dir_isear + 'DATA.csv', delimiter=',')
	# Remove 'No response' row value in isear.csv
	df = df[['Field1','SIT']]

	df = df[~df['SIT'].str.contains('no response')]
	# keep only five emotions (anger, disgust, fear, joy and sadness)
	df = df[~df['Field1'].str.contains('guilt')]
	df = df[~df['Field1'].str.contains('shame')]
	df['labels'] = pd.Categorical(df['Field1']).codes
	classes = len(pd.Categorical(df['labels']).categories)

	train, test = train_test_split(df, test_size=0.2, random_state=0, shuffle=True)

	x_train = np.asarray([preprocessing_isear(sent.lower()) for sent in train['SIT']])
	y_train = np.asarray(train['Field1'])
	x_test = np.asarray([preprocessing_isear(sent.lower())  for sent in test['SIT']])
	y_test = np.asarray(test['Field1'])

	return x_train, y_train, x_test, y_test, classes


def filling_pre_trained_embeddings(num_words, word2idx, word2vec):
	embedding_matrix = np.zeros((num_words, settings.embedding_dim))

	print('embedding_matrix_size: ', np.shape(embedding_matrix))
	count_known = 0
	count_unk = 0

	for word, i in word2idx.items():
		try:
			embedding_vector = word2vec[word]
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
				count_known += 1
			else:
				embedding_matrix[i] = np.random.uniform(-0.25, 0.25, settings.embedding_dim )
				count_unk += 1
		except:
			embedding_matrix[i] = np.random.uniform(-0.25, 0.25, settings.embedding_dim)
			count_unk += 1

	word2vec = None
	print('Embedding loaded words: ', count_known)
	print('Unknown words: ', count_unk)

	return embedding_matrix



def convert_data_one_hot(x_train, y_train, x_test, y_test):
	onehot_encoder = OneHotEncoder(sparse=False)
	y_train = onehot_encoder.fit_transform(y_train.reshape(-1,1))
	y_test = onehot_encoder.fit_transform(y_test.reshape(-1,1))

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(x_train)
	x_train = tokenizer.texts_to_sequences(x_train)
	x_test = tokenizer.texts_to_sequences(x_test)

	# get the word to index mapping for input language
	word2idx = tokenizer.word_index
	print('Found %s unique input tokens.' % len(word2idx))

	# determine maximum length input sequence
	max_len_input = max(len(s) for s in x_train)

	# when padding is not specified it takes the default at the begining of the sentence
	x_train = pad_sequences(x_train, max_len_input, padding='post', truncating='post')
	x_test = pad_sequences(x_test, max_len_input, padding='post', truncating='post')

	return x_train, y_train, x_test, y_test, max_len_input, word2idx



def convert_data(x_train, y_train, x_dev, y_dev, x_test, y_test):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(x_train + x_dev)
	x_train = tokenizer.texts_to_sequences(x_train)
	x_dev = tokenizer.texts_to_sequences(x_dev)
	x_test = tokenizer.texts_to_sequences(x_test)

	y_train = np.array(y_train)
	y_dev = np.array(y_dev)
	y_test = np.array(y_test)

	word2idx = tokenizer.word_index
	print('Found %s unique input tokens.' % len(word2idx))

	# determine maximum length input sequence
	max_len_input = max(len(s) for s in x_train + x_dev)

	x_train = pad_sequences(x_train, max_len_input)#, padding='post', truncating='post')
	x_dev = pad_sequences(x_dev, max_len_input)#, padding='post', truncating='post')
	x_test = pad_sequences(x_test, max_len_input)#, padding='post', truncating='post')

	return x_train, y_train, x_dev, y_dev, x_test, y_test, max_len_input, word2idx