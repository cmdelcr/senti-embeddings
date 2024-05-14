# Refinement of Word Embeddings with Sentiment Content Using Multi-output Neural Networks

This is the Tensorflow 2 implementation of the paper "Refinement of word embeddings with sentiment content using multi-output neural networks". 

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:

<pre>
@article{mcr:2024,
  title={Refinement of Word Embeddings with Sentiment Content Using Multi-output Neural Networks’},
  author={Martín-del-Campo-Rodríguez, C., Batyrshin, Ildar, and Sidorov, Grigori},
  journal={Journal of Intelligent & Fuzzy Systems},
  doi={10.3233/JIFS-219354}
  year={2024}
}
</pre>

The proposed embeddings are available in the URL [senti_embeddings](https://drive.google.com/file/d/1zUO7Hcd1eozNkRUDCaRQBnMhj7T9W8J2/view?usp=sharing)
Once downloaded and decompressed, place the file in results/embeddings to test.



## Generate emotional embeddings
The file run.sh should be executed to generate the embeddings. This file calls the generate_embeddings.py file; it is necessary only to call this file using run.sh since, during the execution, intermediate files are created, these files are required for the correct embedding generation.
The file generate_embeddings.py was used to generate the embeddings. It contains three primary functions: 
- train_multi_output_model: to train the Multi-Output Neural Network. The trained model is saved in results/embeddings.
- get_seti_embeddings: to get the value of the senti-embeddings of all words in the pre-trained embeddings and concatenate the result with the pre-trained embeddings, the resulting embeddings are saved in the temporary file senti_embeddings_tmp.txt
- reduce_senti_emb_pca: to reduce the dimension of the embeddings using IncrementalPCA. The generated embeddings are saved in results/embeddings folder with the name senti_embeddings.txt. The file senti_embeddings_tmp.txt is deleted.


## Test embeddings
The file test_classification.py was used to test the embeddings generated. Using a Bi-LSTM, a classification was made using the dataset Sem-Eval2017, SST-2, and ISEAR, respectively. The specifications for the classification are in the file settings under the commented line Specifications classification. The values resulting from the classification are saved in results/classification folder.


### Folder organization
- **dataset**: contain the datasets ISEAR, SemEval-2017 and SST2 for the test
- **embeddings**: contain the file sources.txt; this file specifies the sources where the pre-trained embeddings of the state-of-the-art can be downloaded. Please download the pre-trained embeddings in this folder
- **lexicon**: contain the three lexica necessary for the training of the model
- **models**: contain python files related to the creation of the model 
- **results**: contain two folders:
	- classification: contains the results of the classification; for each dataset, two files are created, one that specifies the result of the classification for each run and the other where the average and standard deviation are saved. When the file test_classification.py is executed, the new output will be added to the end of each file.
	- embeddings: contains the trained model, as well as the parameters
- **util**: contain python files to read, preprocess and convert input and output files
