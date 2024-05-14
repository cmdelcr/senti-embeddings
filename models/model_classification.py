import os
import statistics
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout, GRU
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

import settings


def create_embedding_layer(embedding_matrix, max_len_input):
	embedding_layer = Embedding(
			embedding_matrix.shape[0],
			embedding_matrix.shape[1],
			weights=[embedding_matrix],
			trainable=False
	)

	return embedding_layer

def create_bilstm_model(embedding_matrix, max_len_input, run, dataset, classes):
	tf.keras.backend.clear_session()
	embedding_layer = create_embedding_layer(embedding_matrix, max_len_input)

	input_ = Input(shape=(max_len_input,))
	x = embedding_layer(input_)
	bidirectional = Bidirectional(LSTM(\
				settings.lstm_dim_sst2 if dataset == 'sst2' else settings.lstm_dim_semeval_isear
				))
	x1 = bidirectional(x)
	output = Dense(classes, 
				kernel_regularizer=regularizers.l2(0.001) if dataset == 'semeval' else None, 
				bias_regularizer=regularizers.l2(0.001) if dataset == 'semeval' else None, 
				activation='sigmoid' if dataset == 'sst2' else 'softmax'
				)(x1)
	model = Model(inputs=input_, outputs=output)
	model.compile('adam',
		'binary_crossentropy' if dataset == 'sst2' else 'categorical_crossentropy', 
		metrics=['accuracy'])

	checkpoint_filepath = settings.dir_out_classification + 'tmp/checkpoint_lstm_' + dataset + '_' + str(run) + '_' + settings.emb_type
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_filepath,
			save_weights_only=True,
			monitor='val_accuracy',
			mode='max',
			save_best_only=True)


	return model, checkpoint_filepath, model_checkpoint_callback



def save_confusion_matrix_multilabel(y_test, pred, run):
	cf_matrix = multilabel_confusion_matrix(y_true=y_test, y_pred=pred)
	cf_matrix = np.array(cf_matrix)
	print('confusion_matrix_shape: ', np.shape(cf_matrix))
	lab, rows, columns = np.shape(cf_matrix)
	path_cf = 'confusion_matrix'
	print(cf_matrix)

	if not os.path.exists(settings.dir_out_classification + path_cf + '_' + settings.emb_type + '.txt'):
		with open(settings.dir_out_classification + path_cf + '_' + settings.emb_type + '.txt', 'w') as file:
			file.close()
	with open(settings.dir_out_classification + path_cf + '_' + settings.emb_type + '.txt', 'a') as file:
		file.write('run: ' + str(run) + '\n')
		for m in range(lab):
			for x in range(rows):
				for y in range(columns):
					file.write(str(cf_matrix[m][x][y]) + (' ' if y < rows-1 else '\n'))
			file.write('\n')
		file.write('\n')
		file.close()


def save_confusion_matrix_binary(y_test, pred, run):
	cf_matrix = confusion_matrix(y_test, pred)
	cf_matrix = np.array(cf_matrix)
	print('confusion_matrix_shape', np.shape(cf_matrix))
	rows, columns = np.shape(cf_matrix)
	path_cf = 'confusion_matrix'
	print(cf_matrix)

	if not os.path.exists(settings.dir_out_classification + path_cf + '_' + settings.emb_type + '.txt'):
		with open(settings.dir_out_classification + path_cf + '_' + settings.emb_type + '.txt', 'w') as file:
			file.close()
	with open(settings.dir_out_classification + path_cf + '_' + settings.emb_type + '.txt', 'a') as file:
		file.write('run: ' + str(run) + '\n')
		for x in range(rows):
			for y in range(columns):
				file.write(str(cf_matrix[x][y]) + (' ' if y < rows-1 else '\n'))
		file.write('\n')
		file.close()


def save_specific_results(acc, precision, recall, f1_macro, f1_micro, run, dataset):
	file_results = 'results_' + dataset + '_in_deep.csv'
	if not os.path.exists(settings.dir_out_classification + file_results):
		with open(settings.dir_out_classification + file_results, 'w') as file:
			file.write('run\tembeddings\taccuracy\tprecision_macro\trecall_macro\tf1_score_macro\tf1_score_micro\n')
			file.close()

	with open(settings.dir_out_classification + file_results, 'a') as file:
		file.write(str(run) + '\t' + settings.emb_type + '\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n' % (acc, precision, recall, f1_macro, f1_micro))
		file.close()




def save_results(arr_acc, arr_precision, arr_recall, arr_f1_macro, arr_f1_micro, dataset):
	file_results = 'results_' + dataset + '.csv'
	if not os.path.exists(settings.dir_out_classification + file_results):
		with open(settings.dir_out_classification + file_results, 'w') as file:
			file.write('embeddings\taccuracy\tprecision_macro\trecall_macro\tf1_score_macro\tf1_score_micro\n')
			file.close()

	with open(settings.dir_out_classification + file_results, 'a') as file:
		file.write(settings.emb_type + '\t%.6f (%.4f)\t%.6f (%.4f)\t%.6f (%.4f)\t%.6f (%.4f)\t%.6f (%.4f)\n' % (statistics.mean(arr_acc), 
			statistics.pstdev(arr_acc), statistics.mean(arr_precision), statistics.pstdev(arr_precision), 
			statistics.mean(arr_recall), statistics.pstdev(arr_recall), statistics.mean(arr_f1_macro), statistics.pstdev(arr_f1_macro),
			statistics.mean(arr_f1_micro), statistics.pstdev(arr_f1_micro)))
		
		file.close()
