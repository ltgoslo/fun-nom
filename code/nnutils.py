from __future__ import unicode_literals
from __future__ import division

import os
import re
import io
import sys
import zipfile
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical #Converts a class vector (integers) to binary class matrix


def form_column(form):
    if form == 'fullform':
        return 1
    elif form == 'lemma':
        return 2

def read_conll_dataset(filepath, form):
    '''
    reads conll-formatted dataset
    returns two dicts: compounds_dict and constituent_dict
    '''
    compounds_dict = {}
    constituent_dict = {}

    f = io.open(filepath, encoding="utf-8")
    compound = ""
    c_number = form_column(form)
    relation = ""
        
    try:
        for i, line in enumerate(f):
            # Comment lines start with #
            if line.startswith("#"):
                continue

            elif len(line) > 2:
                columns = line.split('\t')
                
                # Sometimes CoreNLP returns "Could not handle incoming annotation"
                if len(columns) < 5:
                    continue
                               
                compound += columns[c_number] + ' '
                relation = columns[-1].strip('\n')
                if columns[c_number] not in constituent_dict:
                    constituent_dict[columns[c_number]] = 1
            else:
                compounds_dict[compound.strip(' ')] = relation
                compound = ""
                relation = ""
        f.close()
        return compounds_dict, constituent_dict
                
    except IndexError:
        f.close()
        print(line)


### Embeddings-related functions
def read_embedding(embedding_path):
    # first, build index mapping words in the embeddings set
    # to their embedding vector

    print('Indexing word vectors...')
    embeddings_index = {}
    embeddings_arr = []
    with zipfile.ZipFile(embedding_path) as z:
        for filename in z.namelist():
            if not os.path.isdir(filename) and filename == "model.txt":
                with z.open(filename, 'r') as f:
                    for line in f:
                        try:
                            values = line.split()
                            if sys.version_info[0] == 3:
                                word = values[0].decode()
                            else:
                                word = values[0]
                            coefs = np.asarray(values[1:], dtype='float32')
                            embeddings_index[word] = coefs
                            embeddings_arr.append(coefs)
                        except IndexError:
                            print(line)

    if embeddings_index.get('<unk>') is None:
        print("This embedding model doesn't have a representation for unknown words")
        embeddings_index['<unk>'] = [sum(i)/1000 for i in zip(*embeddings_arr[-1000:])]
    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index, len(embeddings_arr[1])

def get_embedding_vector(embeddings_index, words):
    '''
    Helper function to create an averaged embedding for multi-word constituents
    '''
    word_vectors = []
    for word in words:
        vec = embeddings_index.get(word)
        if vec is None:
            vec = embeddings_index['<unk>']
        word_vectors.append(vec)

    return [sum(i)/len(word_vectors) for i in zip(*word_vectors)]

def prepare_embedding_matrix(embeddings_index, constituent_index, embedding_dim):
    out_of_vocab = 0
    embedding_matrix = np.zeros((len(constituent_index), embedding_dim))
    for word, i in constituent_index.items():
        if len(re.split(' |-', word)) == 1:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                embedding_vector = embeddings_index.get(word.lower())
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                embedding_vector = get_embedding_vector(embeddings_index, re.split(' |-', word))
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # for words that are not part of the embedding dict, use <unk> vector
            out_of_vocab += 1
            embedding_matrix[i] = embeddings_index['<unk>']
    print("%s out-of-vacob words." % (out_of_vocab))
    return embedding_matrix

def prepare_embedding_matrix_c(embeddings_index, c_embedding, constituent_index):
    '''
    Creates embedding matrix based on two embeddings
    '''
    out_of_vocab = 0
    embedding_matrix = np.zeros((len(constituent_index), 600))
    for word, i in constituent_index.items():
        if len(re.split(' |-', word)) == 1:
            embedding_vector = embeddings_index.get(word)
            c_embedding_vector = c_embedding.get(word)
        else:
            embedding_vector = get_embedding_vector(embeddings_index, re.split(' |-', word))
            c_embedding_vector = get_embedding_vector(c_embedding, re.split(' |-', word))
        if embedding_vector is not None and c_embedding_vector is not None:
            embedding_matrix[i] = np.concatenate((embedding_vector, c_embedding_vector))
        elif embedding_vector is not None and c_embedding_vector is None:
            embedding_matrix[i] = np.concatenate((embedding_vector, embedding_vector))
        elif embedding_vector is None and c_embedding_vector is not None:
            embedding_matrix[i] = np.concatenate((c_embedding_vector, c_embedding_vector))
        else:
            # for words that are not part of the embedding dict, use <unk> vector
            out_of_vocab += 1
            embedding_matrix[i] = np.concatenate((embeddings_index['<unk>'], 
                                                  embeddings_index['<unk>']))
    print("%s out-of-vacob words." % (out_of_vocab))
    return embedding_matrix

def encode_labels(labels):
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    #le_name_mapping = dict(zip(labels, encoded_labels))
    # convert integers to dummy variables (i.e. one hot encoded)
    one_hot = to_categorical(encoded_labels)
    return one_hot, encoder

def prepare_traindev_imbalanced(train_path, dev_path, form="fullform", constituent_index=None):
    '''
    Read dev and train data for nombank or pcedt
    '''
    train_dict, constituent_list = read_conll_dataset(train_path, form)
    dev_dict, constituent_list_d = read_conll_dataset(dev_path, form)

    for item in constituent_list_d:
        if item not in constituent_list:
            constituent_list[item] = 1
    
    if constituent_index == None:
        constituent_index = {j:i for i,j in enumerate(constituent_list)}

    x, labels = [], []

    for compound in train_dict:
        cons = compound.split()
        cons1 = ' '.join(cons[:-1])
        cons2 = cons[-1]
        x.append([constituent_index[cons1], constituent_index[cons2]])
        labels.append(train_dict[compound])
    for compound in dev_dict:
        cons = compound.split()
        cons1 = ' '.join(cons[:-1])
        cons2 = cons[-1]
        x.append([constituent_index[cons1], constituent_index[cons2]])
        labels.append(dev_dict[compound])
    encoded_y, encoder = encode_labels(labels)
    x = np.array(x)

    train_y, train_x, dev_y, dev_x = [], [], [], []
    
    train_x = x[:len(train_dict)]
    train_y = encoded_y[:len(train_dict)]


    dev_x = x[len(train_dict):]
    dev_y = encoded_y[len(train_dict):]

    return constituent_index, train_x, train_y, dev_x, dev_y, encoder


def prepare_traindevtest_imbalanced(train_path, dev_path, test_path, form="fullform", constituent_index=None):
    '''
    Read train, dev and test data for nombank or pcedt
    '''
    train_dict, constituent_list = read_conll_dataset(train_path, form)
    dev_dict, constituent_list_d = read_conll_dataset(dev_path, form)
    test_dict, constituent_list_test = read_conll_dataset(test_path, form)

    for item in constituent_list_d:
        if item not in constituent_list:
            constituent_list[item] = 1

    for item in constituent_list_test:
        if item not in constituent_list:
            constituent_list[item] = 1
    
    if constituent_index == None:
        constituent_index = {j:i for i, j in enumerate(constituent_list)}

    x, labels = [], []

    for compound in train_dict:
        cons = compound.split()
        cons1 = ' '.join(cons[:-1])
        cons2 = cons[-1]
        x.append([constituent_index[cons1], constituent_index[cons2]])
        labels.append(train_dict[compound])
    for compound in dev_dict:
        cons = compound.split()
        cons1 = ' '.join(cons[:-1])
        cons2 = cons[-1]
        x.append([constituent_index[cons1], constituent_index[cons2]])
        labels.append(dev_dict[compound])
    for compound in test_dict:
        cons = compound.split()
        cons1 = ' '.join(cons[:-1])
        cons2 = cons[-1]
        x.append([constituent_index[cons1], constituent_index[cons2]])
        labels.append(test_dict[compound])
    encoded_y, encoder = encode_labels(labels)
    x = np.array(x)

    train_y, train_x, dev_y, dev_x, test_y, test_x = [], [], [], [], [], []
    
    train_x = x[:len(train_dict)]
    train_y = encoded_y[:len(train_dict)]


    dev_x = x[len(train_dict):len(train_dict)+len(dev_dict)]
    dev_y = encoded_y[len(train_dict):len(train_dict)+len(dev_dict)]

    test_x = x[len(train_dict)+len(dev_dict):]
    test_y = encoded_y[len(train_dict)+len(dev_dict):]

    return constituent_index, train_x, train_y, dev_x, dev_y, test_x, test_y, encoder
