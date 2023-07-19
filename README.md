# senti-embeddings

This work proposed using a Multi-Output Neural Network to learn emotional embeddings using a combination of lexicons and PCA.


The file should be excecuted to generate the embeddings. This file call the generate_embeddings.py file, it is necessary to only call this file using run.sh  since during the execution intermediate files necessary for the correct embeddings generation.
The file generate_embeddings.py was used to generate the embeddings. It contains three basic functions: train_multi_output_model, to to train the Multi-Output Neural Network; get_seti_embeddings, to get the value of the senti-embeddings of all words in the pre-trained embeddings; and reduce_senti_emb_pca to reduce the dimension of the embeddings using PCA. The generated embeddings are saved in results/embeddings folder with the name senti_embeddings.txt


The file test_classification.py was used to test the embeddings generated. Using a Bi-LSTM, a classification was made using the dataset Sem-Eval2017, SST-2, and ISEAR, respectively. The specifications for the classification are in the file settings under the commented line Specifications classification. The values resulting of the classification are saved in results/classification folder.