#!/bin/bash

python generate_embeddings.py -type 'train'
python generate_embeddings.py -type 'gen_emb'
python generate_embeddings.py -type 'reduce_emb'