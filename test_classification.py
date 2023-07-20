import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import gc
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping

import settings
from util.util_common import read_embeddings
from util.util_class import *
from models.model_classification import *


def init(datasets):
	for dataset in datasets:
		print('\n-----------------------------------------------------')
		print('Dataset: ', dataset)
		print('-----------------------------------------------------')
		if dataset == 'semeval':
			x_train, y_train, x_test, y_test, dict_class = read_sem_eval()
			x_train, y_train, x_test, y_test, max_len_input, word2idx = \
						convert_data_one_hot(x_train, y_train, x_test, y_test)
			classes = 3

		if dataset == 'sst2':
			x_train, y_train, x_dev, y_dev, x_test, y_test = read_sst2()
			x_train, y_train, x_dev, y_dev, x_test, y_test, max_len_input, word2idx = \
						convert_data(x_train, y_train, x_dev, y_dev, x_test, y_test)
			classes = 1
		
		if dataset == 'isear':
			x_train, y_train, x_test, y_test, classes = read_isear()
			x_train, y_train, x_test, y_test, max_len_input, word2idx = \
						convert_data_one_hot(x_train, y_train, x_test, y_test)

		word2vec = read_embeddings(type_emb=settings.emb_type)

		print('Filling pre-trained embeddings...')
		num_words = len(word2idx) + 1
		embedding_matrix = filling_pre_trained_embeddings(num_words, word2idx, word2vec)


		arr_acc = []
		arr_precision = []
		arr_recall = []
		arr_f1_macro = []
		arr_f1_micro = []


		for run in range(1, settings.num_of_runs + 1):
			print('Run: ', run)
			model, checkpoint_filepath, model_checkpoint_callback = create_bilstm_model(\
				embedding_matrix, max_len_input, run, dataset, classes)	
			early_stop = EarlyStopping(monitor='val_loss', patience=10)	

			r = model.fit(x_train, y_train, 
				validation_data=[x_dev, y_dev] if dataset == 'sst2' else None,
				validation_split=0.2 if dataset != 'sst2' else 0.0, 
				batch_size=settings.batch_size, 
				epochs=settings.epochs_semeval if dataset == 'semeval' else \
						settings.epochs_sst2 if dataset == 'sst2' else settings.epochs_isear, 
				verbose=1, 
				callbacks=[model_checkpoint_callback, early_stop]
				)



			if dataset == 'sst2':
				pred = model.predict(x_test, verbose=1)
				pred = np.where(pred > 0.5, 1, 0)
				y_test_ = y_test
			else:
				if dataset == 'semeval':
					# The model weights (that are considered the best) are loaded into the model
					model.load_weights(checkpoint_filepath)
				pred = model.predict(x_test, verbose=1)
				y_test_ = [np.argmax(y, axis=0) for y in y_test]
				pred = [np.argmax(y, axis=0) for y in pred]


			precision = precision_score(y_true=y_test_, y_pred=pred, average='macro')
			recall = recall_score(y_true=y_test_, y_pred=pred, average='macro')
			f1_macro = f1_score(y_true=y_test_, y_pred=pred, average='macro')
			f1_micro = f1_score(y_true=y_test_, y_pred=pred, average='micro')
			acc = accuracy_score(y_true=y_test_, y_pred=pred)


			lstm_dim_vec = 300
			print('acc: ', acc)
			print('f1_micro: ', f1_micro)
			print('f1_macro: ', f1_macro)
			arr_acc.append(acc)
			arr_precision.append(precision)
			arr_recall.append(recall)
			arr_f1_macro.append(f1_macro)
			arr_f1_micro.append(f1_micro)
			
			'''if dataset == 'sst2':
				save_confusion_matrix_binary(y_test_, pred, run)
			else:
				save_confusion_matrix_multilabel(y_test_, pred, run)'''
			save_specific_results(acc, precision, recall, f1_macro, f1_micro, run, dataset)
			
		save_results(arr_acc, arr_precision, arr_recall, arr_f1_macro, arr_f1_micro, dataset)
		gc.collect()



# uncomment lines 119-121 to test with embeddings of the state-of-the-art
# you need to download the necessary pre-trained embeddings first
# embeddings/sources.txt contain the URLs where the pre-trained embeddings can be downloaded
if __name__ == "__main__":
	arr_dir = [settings.dir_senti_embeddings]
	arr_name = ['senti_embeddings']

	#arr_dir = [settings.dir_embeddings_glove, settings.dir_ewe, settings.dir_sawe, settings.dir_sota, \
	#		settings.dir_senti_embeddings]
	#arr_name = ['glove', 'ewe_uni', 'sawe-100', 'sota', 'senti_embeddings']

	for val in range(len(arr_dir)):
		settings.emb_type = arr_name[val]
		settings.path = arr_dir[val]
		init(['semeval', 'sst2', 'isear'])
	


