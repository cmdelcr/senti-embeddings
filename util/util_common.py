import numpy as np
from gensim import models

import settings


def read_embeddings(type_emb='glove', path_test=None):
	print('\n-----------------------------------------')
	print('Loading embeddings ' + type_emb + '...')
	print('-----------------------------------------')

	word2vec = {}
	if type_emb == 'word2vec':
		word2vec = models.KeyedVectors.load_word2vec_format(settings.dir_embeddings_word2vec, binary=True)
		print(settings.dir_embeddings_word2vec)
	else:
		path = settings.dir_embeddings_glove if type_emb == 'glove' else settings.path if path_test is None else path_test
		for line in open(path):
			values = line.split()
			word2vec[str(values[0]).lower()] = np.asarray(values[1:], dtype='float32')
		print(path)

	return word2vec		



