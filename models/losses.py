import tensorflow as tf
from statistics import mean 
import numpy as np

from tensorflow.keras.losses import MeanSquaredError

from sklearn.utils.class_weight import compute_class_weight

from denseweight import DenseWeight


def def_class_weight(y_val, label):
	number_dim = np.shape(y_val)[1]
	weights = np.empty([number_dim, 2])
	for i in range(number_dim):
		weights[i] = compute_class_weight(class_weight='balanced', classes=[0.,1.], y=y_val[:, i])
	
	#return arr_weights
	return weights
	

#weighted categorical cross entropy
class CustomCategoricalCrossEntropy():
	def __init__(self, weights):
		super().__init__()
		self.weights = weights


	def get_weights(self, y_true):
		y_true_ = np.array(y_true)
		number_dim = np.shape(y_true_)[1]
		weights = np.empty([number_dim, 2])
		for i in range(number_dim):
			weights[i] = compute_class_weight(class_weight='balanced', classes=np.unique(y_true_[:, i]), y=y_true_[:, i])

		return weights[:,1]

	def calc_custom_loss(self, y_true, y_pred):
		y_true = tf.cast(y_true, tf.float32)
		y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
		
		# use self.get_weights(..) to obtain weights per batch; self.weights is use for weightening of all the dataset
		loss = tf.reduce_sum(self.get_weights(y_true) * y_true * -tf.math.log(y_pred + tf.keras.backend.epsilon()), axis=-1)
		loss = tf.reduce_mean(loss)
		
		return loss
		

class CustomMSE():
	def __init__(self, weights):
		super().__init__()
		self.weights = weights

	def calc_custom_loss(self, y_true, y_pred):
		mse = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
		y_true = np.array(y_true)
		w = [self.weights[str(y_true[idx][0]) + '_' + str(y_true[idx][1]) + '_' + str(y_true[idx][2])] 
				for idx in range(0, np.shape(y_true)[0])]

		return tf.reduce_mean(mse(y_true, y_pred) * w)

def get_weights(y_val):
	dw = DenseWeight(alpha=1)
	y_val = np.array(y_val)
	weights_valence = dw.fit(y_val[:,0])
	weights_arousal = dw.fit(y_val[:,1])
	weights_dominance = dw.fit(y_val[:,2])

	weights = {}
	for idx in range(np.shape(y_val)[0]):
		weights[str(y_val[idx][0]) + '_'+ str(y_val[idx][1]) + '_'+ str(y_val[idx][2])] =\
		mean([weights_valence[idx], weights_arousal[idx], weights_dominance[idx]])
	
	return weights
