"""
This script is used to train multi-task learning (MTL) model.

Example use:
    python multitask.py -mode nomfun -mtlmode mtle
    -embedding [path to embedding model] --verbose
"""

from __future__ import print_function
from __future__ import division

import sys
import random
import time
from argparse import ArgumentParser
import numpy as np
seed = 123 # fix random seed for reproducibility
random.seed(seed)
np.random.seed(seed)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Input, Flatten, Dropout, Embedding
from nnutils import read_embedding, prepare_embedding_matrix
from nnutils import prepare_traindev_imbalanced, prepare_traindevtest_imbalanced
from transfer import create_old_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

assert sys.version_info > (2,6) and sys.version_info < (3,0), \
       "This code requires python 2.7"

MAX_SEQUENCE_LENGTH = 2

def train_shared(mode, embedding_matrix, num_words, data, ax_data, batch_size,
                epochs, activation_func, optimizer, trainable_embd, dropout,
                mtlmode):
    
    embedding_dim = len(embedding_matrix[0])

    x_train, y_train, x_dev, y_dev = data[0], data[1], data[2], data[3]
    ax_train, ay_train, ax_dev, ay_dev = ax_data[0], ax_data[1], ax_data[2], ax_data[3]

    weights_filepath = '.'.join(["weights_mtl", mode, "b", str(batch_size),
                                mtlmode, "e", str(epochs), activation_func,
                                optimizer, str(trainable_embd), 
                                str(random.randint(0,9999)), str(time.time()), 
                                "hdf5"])
    if mode == 'nomfun':    
        model_checkpoint = ModelCheckpoint(weights_filepath,
                                           monitor='val_output_acc', verbose=1,
                                           mode='auto', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_output_acc', min_delta=0,
                                       patience=5, verbose=0, mode='auto') 
    elif mode == 'funnom':
        model_checkpoint = ModelCheckpoint(weights_filepath,
                                           monitor='val_foutput_acc', verbose=1,
                                           mode='auto', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_foutput_acc', min_delta=0,
                                       patience=5, verbose=0, mode='auto') 

    print("Fine-tuning: %s" % trainable_embd)
    embedding_layer = Embedding(num_words,
                                len(embedding_matrix[0]),
                                trainable=trainable_embd,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                name="embedding")

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name="input")
    embedded_sequences_n = embedding_layer(sequence_input)
    x = Flatten()(embedded_sequences_n)

    if mtlmode == 'mtlf':
        dense = Dense(embedding_dim, activation=activation_func, name="dense_n")(x)
        if dropout > 0:
            dense = Dropout(dropout)(dense)
        npreds = Dense(len(y_train[0]), activation='softmax', name="output")(dense)
        fpreds = Dense(len(ay_train[0]), activation='softmax', name="foutput")(dense)
    elif mtlmode == 'mtle':
        ndense = Dense(embedding_dim, activation=activation_func, name="dense_n")(x)
        fdense = Dense(embedding_dim, activation=activation_func, name="dense_f")(x)
        if dropout:
            ndense = Dropout(dropout)(ndense)
            fdense = Dropout(dropout)(fdense)
        npreds = Dense(len(y_train[0]), activation='softmax', name="output")(ndense)
        fpreds = Dense(len(ay_train[0]), activation='softmax', name="foutput")(fdense)

    model = Model(sequence_input, outputs=[npreds, fpreds])

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  #loss_weights={'output': 1., 'foutput': .1},
                  metrics=['accuracy'])

    history = model.fit({'input': x_train},
                        {'output': y_train, 'foutput': ay_train},
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=([x_dev], [y_dev, ay_dev]),
                        callbacks=[model_checkpoint, early_stopping])

    model_json = model.to_json()
    with open(mode + "-mtl-model.json", "w") as json_file:
        json_file.write(model_json)
    model.summary()
    return model, weights_filepath, history

   
def main():
    argparser = ArgumentParser(description=__doc__)
    argparser.add_argument('-mode', choices=['nomfun', 'funnom'],
                           help='''Mode specifies the main and auxiliary tasks.
                           nomfun: NomBank is main and PCEDT is auxiliary.
                           funnom: PCEDT is main and NomBank is auxiliary''')
    argparser.add_argument('-mtlmode', choices=['mtle', 'mtlf'], default='mtle',
                           help='''Specifies MTL setup: mtle (default) shares
                           the embedding layer only. mtlf shares the
                           embedding and hidden layers.''')
    argparser.add_argument('-embedding', 
                            help='Path to embedding model (zip file!)')
    argparser.add_argument('--test', dest='test', action='store_true',
                            help='Evaluate on the test split.')
    argparser.add_argument('--nofinetune', dest='finetune', action='store_false', 
                            help='Do not update the embedding layer.')
    argparser.add_argument('--batch', type=int, default=5, 
                            help='Batch size. Default: 5')
    argparser.add_argument('--epochs', type=int, default=100, 
                            help='# of epochs. Default: 100')
    argparser.add_argument('--optimizer', choices=['RMSprop', 'adam', 'adamax', 
                                                   'adadelta', 'SGD', 'Adagrad'],
                           default='adam', help='Default: adam')
    argparser.add_argument('--activation', choices=['sigmoid', 'relu'], default='sigmoid',
                           help='Activation function of the hidden layer. Default: sigmoid')
    argparser.add_argument('--dropout', type=float, default=0, 
                           help='Dropout rate. Default: 0')
    argparser.add_argument('--form', type=str, default='fullform', 
                            choices=['fullform', 'lemma'], 
                            help='Token form in compound dataset. Default: fullform')
    argparser.add_argument('--verbose', dest='predict', action='store_true',
                           help='''Write detailed classification report and
                           classification errors to a file''')
    argparser.set_defaults(test=False)
    argparser.set_defaults(finetune=True)
    argparser.set_defaults(predict=False)
    
    args = argparser.parse_args(sys.argv[1:])

    nconstituent_index, ntrain_x, ntrain_y, \
    ndev_x, ndev_y, ntest_x, ntest_y, \
    n_encoder = prepare_traindevtest_imbalanced("./data/nombank_train.conll",
                                                "./data/nombank_dev.conll",
                                                "./data/nombank_test.conll", args.form)

    fconstituent_index, ftrain_x, ftrain_y, \
    fdev_x, fdev_y, ftest_x, ftest_y, \
    f_encoder = prepare_traindevtest_imbalanced("./data/functor_train.conll",
                                                "./data/functor_dev.conll",
                                                "./data/functor_test.conll", args.form)

    print("Loading the embedding model")
    embedding, dimension = read_embedding(args.embedding)

    embedding_index = prepare_embedding_matrix(embedding,
                                               nconstituent_index, dimension)

    m_data = [ntrain_x, ntrain_y, ndev_x, ndev_y]
    ax_data = [ftrain_x, ftrain_y, fdev_x, fdev_y]

    embedding_matrix_copy = embedding_index.copy()

    model, weights_path, _ = train_shared(args.mode, embedding_matrix_copy,
                                       len(nconstituent_index),
                                       m_data, ax_data, args.batch,
                                       args.epochs, args.activation,
                                       args.optimizer,  args.finetune,
                                       args.dropout, args.mtlmode)

    print("Loading best model weights")
    best_model = create_old_model(weights_path, args.mode + "-mtl-model.json")
    best_model.compile(loss='categorical_crossentropy',
                       optimizer=args.optimizer, metrics=['accuracy']) 

    print("Accuracy on dev: %s" % (best_model.evaluate([ndev_x], [ndev_y, fdev_y])))

    if args.test == True:
        n_true = np.argmax(ntest_y, axis=1)
        f_true = np.argmax(ftest_y, axis=1)
        predictions = best_model.predict([ntest_x])
        n_preds = np.argmax(predictions[0], axis=1)
        f_preds = np.argmax(predictions[1], axis=1)
        print("Nombank test accuracy: %f" % (accuracy_score(n_true, n_preds)))
        print("PCEDT test accuracy: %f" % (accuracy_score(f_true, f_preds)))

    if args.predict:
        if args.test == False:
            # write detailed classification report and errors based on the dev
            # split
            ntest_y = ndev_y
            ftest_y = fdev_y
            ftest_x = fdev_x
            ntest_x = ndev_x

        n_true = np.argmax(ntest_y, axis=1)
        n_gold = list(n_encoder.inverse_transform(n_true))

        f_true = np.argmax(ftest_y, axis=1)
        f_gold = list(f_encoder.inverse_transform(f_true))

        predictions = best_model.predict([ntest_x])
        n_preds = np.argmax(predictions[0], axis=1)
        f_preds = np.argmax(predictions[1], axis=1)

        n_pred_classes = list(n_encoder.inverse_transform(n_preds))
        f_pred_classes = list(f_encoder.inverse_transform(f_preds))

        error_file = open('.'.join(weights_path.split('.')[:-1]) + "-errors.txt", 'w')

        error_file.write("Nombank Accuracy\n")
        error_file.write(str(accuracy_score(n_true, n_preds)) + "\n")

        error_file.write("PCEDT Accuracy\n")
        error_file.write(str(accuracy_score(f_true, f_preds)) + "\n")

        error_file.write(classification_report(n_true, n_preds,
                                    labels=list(set(n_true)),
                                    target_names=n_encoder.inverse_transform(list(set(n_true))),
                                    digits=4))
        error_file.write(classification_report(f_true, f_preds,
                                    labels=list(set(f_true)),
                                    target_names=f_encoder.inverse_transform(list(set(f_true))),
                                    digits=4))
        
        error_file.write("\n")
        inv_map = {v: k for k, v in nconstituent_index.iteritems()}
        for i in range(len(n_preds)):
            if n_preds[i] != n_true[i]:
                error_file.write('\t'.join([inv_map[ntest_x[i][0]], 
                                            inv_map[ntest_x[i][1]], 
                                            str(n_pred_classes[i]), 
                                            str(n_gold[i]) + "\n"]))
        error_file.write("\n")
        inv_map = {v: k for k, v in fconstituent_index.iteritems()}
        for i in range(len(f_preds)):
            if f_preds[i] != f_true[i]:
                error_file.write('\t'.join([inv_map[ntest_x[i][0]],
                                            inv_map[ntest_x[i][1]],
                                            str(f_pred_classes[i]),
                                            str(f_gold[i]) + "\n"]))

        error_file.close()

if __name__ == '__main__':
    main()
