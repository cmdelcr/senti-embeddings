import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import gc
import time
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score
from sklearn.metrics import	mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.decomposition import IncrementalPCA


from util.util_common import read_embeddings
from util.util_embeddings import *
from models.model_embeddings import create_compile_model, train_model, save_values_model, save_predictions
from models.model_embeddings import load_pre_trained_multi_out_model
from models.plots import vad_values_statistics, create_save_plots, create_save_roc_curve, create_save_confusion_matrix
import models.losses as losses

import settings



emb_type = settings.emb_type_for_training
dir_name = settings.dir_embeddings_results


def train_multi_output_model():
	not_only_in_vad = False

	print('Loading vad lexicon...')
	dict_vad = read_vad_file()


	print('Loading emo_lex emotions and pos_neg...')
	dict_pos_neg, arr_emotions_pos_neg = read_emo_lex_file(dict_vad, not_only_in_vad)

	print('Loading subjectivity_clues lexicon...')
	dict_sub = read_subjectivity_clues(dict_vad, not_only_in_vad)

	print('Combining emo_lex_pos_neg and sub_clues...')
	dict_comb = combined_values_pos_neg(dict_pos_neg, dict_sub, dict_vad)
	
	print(dict_comb['terrorism'])
	print(dict_vad['terrorism'])
	exit()


	word2vec = read_embeddings(type_emb=emb_type)
	word2idx, vocabulary = getting_idx_word(emb_type, dict_vad, word2vec)
	embedding_matrix, vocabulary, y_vad, y_pos_neg = filling_embeddings(
				word2idx, word2vec, vocabulary, dict_vad, dict_comb)

	print('size_embeddings: ', np.shape(embedding_matrix))

	vad_values_statistics(dir_name, embedding_matrix, y_vad)

	dict_labels = {}
	for idx, word in enumerate(vocabulary):
		dict_labels[idx] = word
	index_label = list(dict_labels.keys())

	x_train, x_dev, y_train_vad, y_dev_vad, y_train_pos_neg, y_dev_pos_neg, idx_train, idx_dev = \
				train_test_split(
					embedding_matrix, y_vad, y_pos_neg, index_label, test_size=0.1, random_state=0, 
					shuffle=True)

	print('--------')
	print('negative, positive, neutral')
	print('train: ', np.sum(y_train_pos_neg, axis=0))
	print('valid: ', np.sum(y_dev_pos_neg, axis=0))
	print('--------\n')


	print('Defining weights...')
	class_weights_pos_neg = losses.def_class_weight(np.concatenate((y_train_pos_neg, y_dev_pos_neg), axis=0), 'pos_neg')
	weights_vad = losses.get_weights(np.concatenate((y_train_vad, y_dev_vad), axis=0))


	print('--------------------------')
	print('Creating the model...')
	model = create_compile_model(dir_name, len(embedding_matrix[0]), len(y_train_pos_neg[0]), 
					weights_vad, class_weights_pos_neg)
	r = train_model(model, x_train, y_train_vad, y_train_pos_neg, x_dev, y_dev_vad, y_dev_pos_neg)

	save_values_model(dir_name, x_train, y_train_vad, y_train_pos_neg, x_dev,
		y_dev_vad, y_dev_pos_neg, idx_train, idx_dev, dict_labels)


	model.save(dir_name + 'model')

	create_save_plots(dir_name, r)

	pred_reg, pred_class_pos_neg = model.predict(x_dev)

	mse = mean_squared_error(y_dev_vad, pred_reg)
	mae = mean_absolute_error(y_dev_vad, pred_reg)
	r2 = r2_score(y_dev_vad, pred_reg)
	print('------------------------------------')
	print('------------------------------------')
	print('------------------------------------')
	print('regression vad...')
	print('mse', mse)
	print('mae', mae)
	print('r2: ', r2)

	print('------------------------------------')
	print('classification pos_neg...')
	pred_class_pos_neg = pred_class_pos_neg.round()	
	acc = accuracy_score(y_dev_pos_neg, y_pred=pred_class_pos_neg)
	precision = precision_score(y_dev_pos_neg, y_pred=pred_class_pos_neg, average='macro')
	recall = recall_score(y_dev_pos_neg, y_pred=pred_class_pos_neg, average='macro')
	f1 = f1_score(y_dev_pos_neg, y_pred=pred_class_pos_neg, average='macro')
	roc_auc = roc_auc_score(y_dev_pos_neg, pred_class_pos_neg, average='macro')
	print('accuracy: ', acc)
	print('precision: ', precision)
	print('recall: ', recall)
	print('f1: ', f1)
	print('roc_auc: ', roc_auc)

	create_save_roc_curve(dir_name, y_dev_pos_neg, pred_class_pos_neg, 'pos_neg', arr_emotions_pos_neg)
	create_save_confusion_matrix(dir_name, y_dev_pos_neg, pred_class_pos_neg, 'pos_neg', arr_emotions_pos_neg)
	save_predictions(dir_name, mse, mae, r2, acc, precision, recall, f1, roc_auc)
	gc.collect()



def get_seti_embeddings():
	print('########################################################')
	print('Starting get_senti_embeddings')
	word2vec = read_embeddings(type_emb=emb_type)
	vocabulary = list(word2vec.keys())
	x_emb = np.zeros((len(vocabulary), 300))
	for idx, word in enumerate(vocabulary):
		x_emb[idx] = word2vec[word]
	word2vec = None

	model = load_pre_trained_multi_out_model(dir_name)
	with open(dir_name + settings.senti_emb_aux + '.txt', 'w') as f:
		f.close()

	add_val = 100000
	for i in range(0, len(x_emb), add_val):
		top_range = i + add_val
		if top_range > len(x_emb):
			pred = model.predict(x_emb[i:len(x_emb)])
			print('chuck: ', i, ' - ', len(x_emb))
		else:
			pred = model.predict(x_emb[i:i+add_val])
			print('chuck: ', i, ' - ', i+add_val)
		print('pred_size: ', np.shape(pred))

		size_saved_matrix = 0
		with open(dir_name + settings.senti_emb_aux + '.txt', 'a') as f:
			mat = np.matrix(pred)
			k = 0
			for w_vec in mat:
				f.write(vocabulary[i+k] + " ")
				np.savetxt(f, fmt='%.6f', X=w_vec)
				size_saved_matrix += 1
				k += 1
			f.close()
		print('size_embeddings: ', size_saved_matrix)
	gc.collect()


def reduce_senti_emb_pca():
	print('############################################################')
	print('Starting to reduce with PCA')
	path = dir_name + settings.senti_emb_aux + '.txt'
	num_lines = sum(1 for _ in open(path))
	ipca =reduce_dim_embeddings(num_lines, path)

	with open(dir_name + settings.name_final_senti_emb + '.txt', 'w') as f:
		f.close()


	chunk_size = 10000
	for i in range(0, num_lines, chunk_size):
		vocabulary = []
		senti_embedding = []
		top_range = i + chunk_size
		if top_range > num_lines:
			with open(path, "r") as text_file:
				for line in itertools.islice(text_file, i, num_lines):
					values = line.split()
					vocabulary.append(str(values[0]).lower())
					senti_embedding.append(np.asarray(values[1:], dtype='float32'))
			print('chuck: ', i, ' - ', num_lines)
		else:
			with open(path, "r") as text_file:
				for line in itertools.islice(text_file, i, i+chunk_size):
					values = line.split()
					vocabulary.append(str(values[0]).lower())
					senti_embedding.append(np.asarray(values[1:], dtype='float32'))
			print('chuck: ', i, ' - ', i+chunk_size)	
		senti_embedding = np.array(senti_embedding)
		senti_embedding = ipca.transform(senti_embedding)
		print('pred_size: ', np.shape(senti_embedding))


		size_saved_matrix = 0
		with open(dir_name + settings.name_final_senti_emb + '.txt', 'a') as f:
			mat = np.matrix(senti_embedding)
			k = 0
			for w_vec in mat:
				f.write(vocabulary[k].replace(" ", "_" ) + " ")
				np.savetxt(f, fmt='%.6f', X=w_vec)
				size_saved_matrix += 1
				k += 1
			f.close()
		print('size_embeddings: ', size_saved_matrix)
	os.remove(path)


types_run = ['train', 'gen_emb', 'reduce_emb']
if __name__ == "__main__":
	try:
		input_type = sys.argv[sys.argv.index("-type")+1]
	except ValueError as e:
		print("The call to the program does not contain the option -type")
		exit(1)
	if input_type == types_run[0]:
		train_multi_output_model()
	elif input_type == types_run[1]:
		get_seti_embeddings()
	elif input_type == types_run[2]:
		reduce_senti_emb_pca()
	else:
		print("Types to call the program are: ")
		print("\t- 'train' to train the network")
		print("\t- 'gen_emb' to generate the embeddings from the trained network and concatenate the pre-trained embeddings")
		print("\t- 'reduce_emb' to reduce the embeddings dimention to 300")
		exit(1)

