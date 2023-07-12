import os
import re
import random
import itertools
import collections
import pandas as pd
import numpy as np

from gensim import models

from sklearn.decomposition import IncrementalPCA

from nltk.stem import WordNetLemmatizer

import settings



########################################################################################################################
# Functions for reading lexicons
########################################################################################################################

def read_vad_file():
	dict_data = {}
	df = pd.read_csv(settings.dir_vad, 
				keep_default_na=False, header=None, sep='\t')
	for index, row in df.iterrows(): #V, A, D
		dict_data[str(row[0]).lower()] = [float(row[1]), float(row[2]), float(row[3])]
	print('size VAD: ', len(dict_data))

	return dict_data



# modifies the last column that represent other_emotion (for those words where is no emotion assigned)
def verify_emo_pol(dict_emo_lex, label):
	arr_remove = []
	for key, value in dict_emo_lex.items():
		if not value.any():
			value[-1] = 1
			dict_emo_lex[key] = value

		counter = collections.Counter(dict_emo_lex[key])
		if counter[1] > 1:
			arr_remove.append(key)
	
	for key in arr_remove:
		del dict_emo_lex[key]

	print('Size ' + label +': ', len(dict_emo_lex))

	return dict_emo_lex



def read_emo_lex_file(dict_vad_, not_only_in_vad):
	df_emo_lex = pd.read_csv(settings.dir_nrc, 
			keep_default_na=False, header=None, sep='\t')

	arr_emotions_pos_neg = ['negative', 'positive', 'neutral']

	dict_emo_lex = {}
	count = 0

	dict_pos_neg = {}
	count_pos_neg = 0

	count_values_vad = 0

	dict_count_pos_neg= {}
	for index, row in df_emo_lex.iterrows():
		dict_count_pos_neg[str(row[0])] = 1
		if str(row[0]) in dict_vad_ or not_only_in_vad:
			if str(row[1]) == 'negative' or str(row[1]) == 'positive':
				if str(row[0]) in dict_pos_neg:
					arr_pos_neg = dict_pos_neg[str(row[0])]
				else:
					arr_pos_neg = np.zeros(3)

				idx = arr_emotions_pos_neg.index(str(row[1]))
				if int(row[2]) == 1 and len(np.where(arr_pos_neg == 1)[0]) > 0:
					count_pos_neg += 1

				arr_pos_neg[idx] = int(row[2])
				dict_pos_neg[str(row[0])] = arr_pos_neg
		else:
			count_values_vad += 1


	print('Total number of words: ', len(dict_count_pos_neg))
	print(len(dict_pos_neg), ', words_with two values: ', count_pos_neg)
	print('Num words ignored (not in vad)', count_values_vad)
	final_dict_pos_neg = verify_emo_pol(dict_pos_neg, 'pos_neg')

	return final_dict_pos_neg, arr_emotions_pos_neg



def def_value(row):
	arr_value = np.zeros(4)

	# strongly_positive, weakly_positive, strongly_negative, and weakly_negative
	if 'positive' in str(row[5]):
		if 'strong' in str(row[0]):
			arr_value[0] = 1
		else:
			arr_value[1] = 1
	else:
		if 'strong' in str(row[0]):
			arr_value[2] = 1
		else:
			arr_value[3] = 1

	return arr_value



def read_subjectivity_clues(dict_vad, not_only_in_vad):
	dict_data = {}
	count = 0

	with open(settings.dir_sub_clues, 'r') as file:
		count_values_vad = 0
		for line in file:
			row = line.split()
			key = re.sub(r'word1=', '', str(row[2])).lower()
			if key in dict_vad or not_only_in_vad:
				if key in dict_data:
					# strongly_positive, weakly_positive, strongly_negative, and weakly_negative
					if np.where(def_value(row) == 1)[0][0] in np.where(dict_data[key] == 1)[0]:
						continue

					arr_sub = dict_data[key] + def_value(row)
					count += 1
				else:
					arr_sub = def_value(row)

				dict_data[key] = arr_sub
			else:
				count_values_vad += 1
		file.close()
	print(len(dict_data), ', words_with two values: ', count)
	print('Num words ignored (not in vad)', count_values_vad)
	
	return dict_data#, arr_counts



# Add to the vocaburaly the lemmas from the pre-trained embeddings
def getting_idx_word(emb_type, voc_, word2vec):
	print('Getting lemmas...')
	counter_lem_lex = 0
	counter_word = 0
	counter_only_in_vad = 0
	counter_word_dict = 0
	word2idx = {}
	vocabulary = []


	lemmatizer = WordNetLemmatizer()
	list_keys = list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys())
	words_lexicons = dict.fromkeys(voc_, 0)
	print('Num words in lexico: ', len(words_lexicons))


	for key in list_keys:
		if key in words_lexicons:
			vocabulary.append(key)
			counter_word_dict += 1
			word2idx[key] = counter_word
			counter_word += 1
		else:
			lemma = lemmatizer.lemmatize(key)
			if lemma in words_lexicons and lemma not in word2idx:
				counter_lem_lex += 1
				vocabulary.append(key)
				word2idx[key] = counter_word
				counter_word += 1

	set_uniq_vad = list(set(words_lexicons).difference(set(vocabulary)))
	for value in set_uniq_vad:
		counter_only_in_vad += 1
		vocabulary.append(value)
		word2idx[value] = counter_word
		counter_word += 1


	print('words in lexicon and ' + emb_type + ': ', counter_word_dict)
	print('lemmas in lexicon: ', counter_lem_lex)
	print('words only in lexico: ', counter_only_in_vad)
	print('final vocabulary size: ', counter_word)

	return word2idx, vocabulary


# return array with size 3 and only one value with 1:  ([1, 0, 0]) equivalent to (['negative', 'positive', 'neutral'])
def aux_ver_pol(arr):
	arr_aux = np.zeros(3)
	#strongly_positive and weakly_positive : positive
	if arr[0] == 1 and arr[1] == 1:
		arr_aux[1] = 1
	#strongly_negative anf  weakly_negative : negative
	elif arr[2] == 1 and arr[3] == 1:
		arr_aux[0] = 1
	#weakly_positive and weakly_negative : neutral
	elif arr[1] == 1 and arr[3] == 1:
		arr_aux[2] = 1
	#strongly_positive and weakly_negative : positive
	elif arr[0] == 1 and arr[3] == 1:
		arr_aux[1] = 1
	#strongly_positive and strongly_negative : neutral
	elif arr[0] == 1 and arr[2] == 1:
		arr_aux[2] = 1
	#weakly_positive and strongly_negative : negative
	elif arr[1] == 1 and arr[2] == 1:
		arr_aux[0] = 1
	
	return arr_aux


# if val have more than one non-zero value, return an array according the the restriction of the function aux_ver_pol
# else, if the value of the array is strong_positive or weakly_positive, return positive ([0, 1, 0])
# if the value of the array is strong_negative or weakly_negative, return negative ([1, 0, 0])
def init_arr_sub(val):
	counter = collections.Counter(val)
	if counter[1] > 1:
		arr_sub = aux_ver_pol(val)
	else:
		arr_sub = np.zeros(3)
		if val[0] == 1 or val[1] == 1:
			arr_sub[1] = 1
		else:
			arr_sub[0] = 1

	return arr_sub


# Combine word from emo-lex and ubjectivity_clues
# modify the values in subjectivity clues acccording to the functions aux_ver_pol and init_arr_sub
# return a dict with the combined vocabulary and the values correcpoding to (negative, positive, neutral)
def combined_values_pos_neg(dict_pos_neg, dict_sub, dict_vad):
	vocab = list(set(list(dict_pos_neg)).union(set(list(dict_sub))))
	print('-------------------------------------------------')
	print('Pos_neg: ', len(dict_pos_neg))
	print('negative', 'positive', 'neutral')
	aux = np.array(list(dict_pos_neg.values()))
	print(np.sum(aux, axis=0))
	print('--------')
	print('Sub_clues: ', len(dict_sub))
	print('strongly_positive', 'weakly_positive', 'strongly_negative', 'weakly_negative')
	aux = np.array(list(dict_sub.values()))
	print(np.sum(aux, axis=0))

	dict_comb = {}
	for word in vocab:
		if word in dict_sub:
			arr_sub = init_arr_sub(dict_sub[word])
		if word in dict_pos_neg and word in dict_sub:
			if np.where(dict_pos_neg[word] == 1)[0][0] == np.where(arr_sub == 1)[0][0]:
				dict_comb[word] = arr_sub
			else:
				if np.where(dict_pos_neg[word] == 1)[0][0] != 2:
					dict_comb[word] = dict_pos_neg[word]
				else:
					dict_comb[word] = arr_sub
		else:
			if word in dict_pos_neg:
				dict_comb[word] = dict_pos_neg[word]
			else:
				dict_comb[word] = arr_sub

	print('--------')
	print('Combined lexicons: ', len(dict_comb))
	print('negative', 'positive', 'neutral')
	aux = np.array(list(dict_comb.values()))
	print(np.sum(aux, axis=0))
	print('-------------------------------------------------')

	return dict_comb


# For each word in the vocabulary, assingn two array representing VAD values (size 3 with continues values) and 
# polarity values (size 3 with discrete values)
def filling_embeddings(word2idx, word2vec, embeddings_list, dict_vad, dict_pos_neg):
	print('-----------------------------------------')
	print('***Filling pre-trained embeddings...')
	count_known_words = 0
	count_known_lemmas = 0
	count_unknown_words = 0

	count_vad = 0
	count_vad_lemma = 0
	y_vad = []
	count_pos_neg = 0
	count_pos_neg_lemma = 0
	y_pos_neg = []


	lemmatizer = WordNetLemmatizer()
	embedding_matrix = np.zeros((len(embeddings_list), 300))
	print('embedding_matrix: ', np.shape(embedding_matrix))
	size_output_pos_neg = random.choice(list(dict_pos_neg.items()))[1]

	i = 0
	for word in embeddings_list:
		try:
			embedding_vector = word2vec[word]
			embedding_matrix[i] = embedding_vector
			count_known_words += 1
		except:
			lemma = lemmatizer.lemmatize(word)
			try:
				embedding_vector = word2vec[lemma]
				embedding_matrix[i] = embedding_vector
				count_known_lemmas += 1
			except:
				# words not found in embedding index will be initialized with a gaussian distribution.
				embedding_matrix[i] = np.random.uniform(-0.25, 0.25, 300)
				count_unknown_words += 1

		lemma = lemmatizer.lemmatize(word)
		# addings vad values
		if word in dict_vad:
			y_vad.append(dict_vad[word])
			count_vad += 1
		else:
			if lemma in dict_vad:
				y_vad.append(dict_vad[lemma])
				count_vad_lemma += 1
			else:
				y_vad.append([0.5, 0.5, 0.5])

		# addings pos_neg values
		if word in dict_pos_neg:
			y_pos_neg.append(dict_pos_neg[word])
			count_pos_neg += 1
		else:
			if lemma in dict_pos_neg:
				y_pos_neg.append(dict_pos_neg[lemma])
				count_pos_neg_lemma += 1
			else:
				arr = np.zeros(len(size_output_pos_neg))
				arr[-1] = 1
				y_pos_neg.append(arr)

		i += 1

	y_vad = np.asarray(y_vad, dtype='float32')
	y_pos_neg = np.asarray(y_pos_neg, dtype='int32')

	print('Size words initialized with a gaussian distribution: ', count_unknown_words)
	print('Size words with a value in word2vec: ', count_known_words)
	print('Num of vad values: ', count_vad, ', size: ', np.shape(y_vad))
	print('Num of vad lemma values: ', count_vad_lemma, ', size: ', np.shape(y_vad))
	print('Num pos_neg: ', count_pos_neg, ', size: ', np.shape(y_pos_neg))
	print('Num pos_neg lemma: ', count_pos_neg_lemma, ', size: ', np.shape(y_pos_neg))

	return embedding_matrix, embeddings_list, y_vad, y_pos_neg



def get_values_model(dir_):
	embedding_matrix = np.load(dir_ + 'embedding_matrix.npy')
	#y_vad = np.load(dir_ + 'y_vad')
	#y_emo_lex = np.load(dir_ + 'y_emo_lex')
	with open(dir_ + 'voc_.txt', 'r') as file:
		voc_ = file.read().split(',')
		file.close()
	with open(dir_ + 'voc.txt', 'r') as file:
		voc = file.read().split(',')
		file.close()
	
	return embedding_matrix, voc_, voc



def reduce_dim_embeddings(num_lines, path):
	print('initializing pca')
	ipca = IncrementalPCA(n_components=300)

	print('loadding embeddings')
	senti_embedding = []
	n = num_lines # how many rows we have in the dataset
	chunk_size = 100000 # how many rows we feed to IPCA at a time, the divisor of n
	for i in range(0, n, chunk_size):
		senti_embedding = []
		top_range = i + chunk_size
		if top_range > n:
			with open(path, "r") as text_file:
				for line in itertools.islice(text_file, i, n):
					values = line.split()
					senti_embedding.append(np.asarray(values[1:], dtype='float32'))
			print('chuck: ', i, ' - ', n)
		else:
			with open(path, "r") as text_file:
				for line in itertools.islice(text_file, i, i+chunk_size):
					values = line.split()
					senti_embedding.append(np.asarray(values[1:], dtype='float32'))
			print('chuck: ', i, ' - ', i+chunk_size)	
		print('len_senti_emb: ', len(senti_embedding))
		senti_embedding = np.array(senti_embedding)
		ipca.partial_fit(senti_embedding)

	print('shape_embeddings: ', np.shape(senti_embedding))
	print('starting pca')

	return ipca
