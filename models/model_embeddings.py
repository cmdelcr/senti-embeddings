import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import time
import itertools

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import CategoricalAccuracy, RootMeanSquaredError
from tensorflow.keras.utils import plot_model

from sklearn.decomposition import IncrementalPCA

from . import losses 
import settings


def create_compile_model(dir_name, input_shape, output_size_pos_neg, weights_vad,
				class_weights_pos_neg):

	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

	input_ = Input(shape=(input_shape,), name='input_layer')
	hidden_shared_layer = Dense(180, name='hidden_shared_layer', #kernel_initializer='he_normal',
		kernel_regularizer=regularizers.l2(0.001)
		)	
	x_shared_layer = hidden_shared_layer(input_)
	active_layer = LeakyReLU(0.3)
	x_shared_layer = active_layer(x_shared_layer)
	dropout_in = Dropout(0.5)
	x_shared_layer_pos = dropout_in(x_shared_layer)
	

	#layer regression_vad
	hidden_layer_vad = Dense(180, name='hidden_layer_vad_1',# activation='tanh',
		kernel_regularizer=regularizers.l2(0.008), 
		kernel_initializer='he_normal'
		)
	x_vad = hidden_layer_vad(x_shared_layer)
	active_layer_vad = LeakyReLU(0.3, name='leakyrelu_vad')
	x_vad = active_layer_vad(x_vad)


	#layer classification_pos_neg
	hidden_layer_pos_eng = Dense(150, name='hidden_layer_pos_neg_1',# activation='tanh', 
		kernel_regularizer=regularizers.l2(0.008),
		kernel_initializer='he_normal'
		)
	x_pos_neg = hidden_layer_pos_eng(x_shared_layer_pos)
	active_layer_pos_neg1 = LeakyReLU(0.2, name='leakyrelu_pos_neg1')
	x_pos_neg = active_layer_pos_neg1(x_pos_neg)
	dropout = Dropout(0.5)
	x_pos_neg = dropout(x_pos_neg)
	

	output_regression = Dense(3, activation='linear', name='output_reg_vad')(x_vad)
	output_class_pos_neg = Dense(output_size_pos_neg, activation='softmax', name='output_class_pos_neg')(x_pos_neg)

	
	model = Model(inputs=[input_], 
		outputs=[output_regression, output_class_pos_neg])
	
	model.compile(
			loss=[losses.CustomMSE(weights_vad).calc_custom_loss,
				  losses.CustomCategoricalCrossEntropy(class_weights_pos_neg).calc_custom_loss],
			optimizer=Adam(learning_rate=0.001), # SGD(learning_rate=0.01), 
			run_eagerly=True,
			metrics=[RootMeanSquaredError(), 
					CategoricalAccuracy(name='acc_pos_neg')] 
	)
	model.summary()
	plot_model(model, to_file=dir_name + 'model.png', show_shapes=True, show_layer_names=True)


	return model


class PrinterCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		print('Epoch: {}, loss: {:0.6f}, val Loss: {:0.6f}, lr: {:0.6f}\nmse: {:0.6f}, val_mse: {:0.6f}\nacc_pos_neg: {:0.6f}, val_acc_pos_neg: {:0.6f}, '.format(epoch+1,
			   logs['loss'],
			   logs['val_loss'],
			   logs['lr'],
			   logs['output_reg_vad_loss'],
			   logs['val_output_reg_vad_loss'],
			   logs['output_class_pos_neg_acc_pos_neg'],
			   logs['val_output_class_pos_neg_acc_pos_neg']))

	def on_epoch_begin(self, epoch, logs=None):
		print('-'*50)
		print('Epoch: {}/{}'.format(epoch+1, settings.epochs))

def decay_schedule(epoch, lr):
	# decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
	if (epoch % 20 == 0) and (epoch != 0):
		lr = lr / 2
		#print('lr:', lr)
	return lr


def train_model(model, x_train, y_train_vad, y_train_pos_neg, x_dev, y_dev_vad, y_dev_pos_neg):
	print('Training model...')
	lr_scheduler = LearningRateScheduler(decay_schedule)

	r = model.fit(x_train, [y_train_vad, y_train_pos_neg], 
		batch_size=settings.batch_size_emo, 
		epochs=settings.epochs, 
		callbacks=[lr_scheduler, PrinterCallback()],
		validation_data=(x_dev, [y_dev_vad, y_dev_pos_neg]),
		verbose=0)

	return r




def save_values_model(dir_name, x_train, y_train_vad, y_train_pos_neg,
				x_dev, y_dev_vad, y_dev_pos_neg, idx_train, idx_dev, dict_voc):
	dir_ = dir_name + 'parameters/'
	if not os.path.exists(dir_):
		os.makedirs(dir_)
	np.save(dir_ + 'x_train', x_train)
	np.save(dir_ + 'y_train_vad', y_train_vad)
	np.save(dir_ + 'y_train_pos_neg', y_train_pos_neg)
	np.save(dir_ + 'x_dev', x_dev)
	np.save(dir_ + 'y_dev_vad', y_dev_vad)
	np.save(dir_ + 'y_dev_pos_neg', y_dev_pos_neg)
	np.save(dir_ + 'idx_train', idx_train)
	np.save(dir_ + 'idx_dev', idx_dev)

	with open(dir_ + 'voc.txt', 'w') as file:
		for key, value in dict_voc.items():
			file.write(str(key) + '|' + str(value) + '\n')
		file.close()


def save_predictions(dir_name, mse, mae, r2, acc, precision, recall, f1, roc_auc):
	with open(dir_name + 'figures/results_training.txt', 'w') as file:
		file.write('------------------------------------\n')
		file.write('regression vad...\n')
		file.write('mse: ' + str(mse) + '\n')
		file.write('mae: ' + str(mae) + '\n')
		file.write('r2: ' + str(r2) + '\n')
		file.write('------------------------------------\n')
		file.write('classification pos_neg...\n')
		file.write('accuracy: ' + str(acc) + '\n')
		file.write('precision: ' + str(precision) + '\n')
		file.write('recall: ' + str(recall) + '\n')
		file.write('f1: ' + str(f1) + '\n')
		file.write('roc_auc: ' + str(roc_auc) + '\n')



def load_pre_trained_multi_out_model(dir_name):
	print('\n\n##########################################################################')
	print('Loading pre-trained multi-output model...')
	model = tf.keras.models.load_model(dir_name + 'model', compile=False)
	# Auxiliary dictionary to describe the network graph
	network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

	# Set the input layers of each layer
	for layer in model.layers:
		for node in layer._outbound_nodes:
			layer_name = node.outbound_layer.name
			if layer_name not in network_dict['input_layers_of']:
				network_dict['input_layers_of'].update(
						{layer_name: [layer.name]})
			else:
				network_dict['input_layers_of'][layer_name].append(layer.name)
	
	# Set the output tensor of the input layer
	network_dict['new_output_tensor_of'].update(
			{model.layers[0].name: model.input})
	#print(network_dict['input_layers_of'])
	#print(network_dict['new_output_tensor_of'])
	#print('------------------------')

	# Iterate over all layers after the input
	model_outputs = []
	concat = tf.keras.layers.Concatenate(name='concat')
	concat_1 = tf.keras.layers.Concatenate(name='concat_1')

	for layer in model.layers[1:]:
		# Insert layer if name matches the regular expression
		if layer.name in ['output_reg_vad', 'output_class_pos_neg']:
			continue
		
		# Determine input tensors
		layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
				for layer_aux in network_dict['input_layers_of'][layer.name]]

		if len(layer_input) == 1:
			layer_input = layer_input[0]

		x = layer(layer_input)
		
		# Set new output tensor (the original one, or the one of the inserted layer)
		network_dict['new_output_tensor_of'].update({layer.name: x})

		# Save tensor in output list if it is output in initial model
		if layer.name in ['leakyrelu_vad', 'leakyrelu_pos_neg1']:
			model_outputs.append(x)
	out = concat(model_outputs)
	out1 = concat_1([model.input, out])

	model_ = Model(inputs=model.inputs, outputs=out1)
	model_.summary()
	#plot_model(model_, to_file=dir_name + 'model_emb_1.png', show_shapes=True, show_layer_names=True)

	return model_

