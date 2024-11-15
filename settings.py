

dir_embeddings_word2vec = 'embeddings/GoogleNews-vectors-negative300.bin'
dir_embeddings_glove = 'embeddings/glove.42B.300d.txt'
dir_ewe = 'embeddings/ewe_uni.txt'
dir_sawe = 'embeddings/sawe-tanh-pca-100-glove.txt'
dir_sota = 'embeddings/senti_embeddings_sota.txt'
dir_senti_embeddings = 'results/embeddings/senti_embeddings.txt'


dir_semeval = 'datasets/semeval/'
dir_sst2 = 'datasets/sst2/'
dir_isear = 'datasets/isear/'


dir_vad = 'lexicon/NRC-VAD-Lexicon.txt'
dir_sub_clues = 'lexicon/subjclueslen1-HLTEMNLP05.tff'
dir_nrc = 'lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'


dir_out_classification = 'results/classification/'



#####################################################################
# Specifications embeddings
epochs = 200
batch_size_emo = 128
dir_embeddings_results = 'results/embeddings/'
# emb_type_for_training are the pre-trained embeddings using to train the multi.output neural network
# possible values: glove, word2vec, (to use other embeddings in txt file (word value_embedding), use any other name)
emb_type_for_training = 'glove'
senti_emb_aux = 'senti_embeddings_tmp'
name_final_senti_emb = 'senti_embeddings'


#####################################################################
# Specifications classification
num_of_runs = 10
emb_type = 'glove'
path = dir_embeddings_glove 
embedding_dim = 300

lstm_dim_semeval_isear = 150
batch_size = 128
epochs_semeval = 20
epochs_sst2 = 10
epochs_isear = 30
lstm_dim_sst2 = 50
