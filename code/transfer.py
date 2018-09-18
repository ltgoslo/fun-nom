"""
This script can be used to train both single-task learning (STL) models and
transfer learning (TL) models.

STL training example:
    python transfer.py -mode nombank -train data/nombank_train.conll 
    -dev data/nombank_dev.conll --test data/nombank_test.conll 
    -embedding [path to embedding model] --verbose 

Transfer learning example:
    python transfer.py -mode nombank -train data/nombank_train.conll 
    -dev data/nombank_dev.conll --test data/nombank_test.conll 
    -embedding [path to embedding model] --verbose --transfer e
    --arch pcedt-model.json --weights [path to the aux model weights]
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import sys
import random
import time
import copy
from argparse import ArgumentParser
import numpy as np
seed = 123 # fixed random seed for reproducibility
random.seed(seed)
np.random.seed(seed)
from keras.layers import Dense, Input, Flatten, Dropout, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential, model_from_json
from nnutils import read_embedding, prepare_embedding_matrix
from nnutils import prepare_traindev_imbalanced, prepare_traindevtest_imbalanced
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

assert sys.version_info > (2,6) and sys.version_info < (3,0), \
       "This code requires python 2.7"


MAX_SEQUENCE_LENGTH = 2

def create_old_model(weights_path, architecture_path):
    '''
    Creates a model with the old architecture and load its weights
    '''
    model = model_from_json(open(architecture_path, 'r').read())
    model.load_weights(weights_path)

    return model

def load_weights(new_model, weights_path, architecture_path, transfer_mode):
    '''
    Set the weights in the new model
    '''
    old_model = create_old_model(weights_path, architecture_path)
    
    layer_dict = dict([(layer.name, layer) for layer in old_model.layers])

    if transfer_mode == "e":
        print("Setting weights for the embedding layer")
        new_model.layers[1].set_weights(layer_dict['embedding'].get_weights())
        
    elif transfer_mode == "h":
        print("Setting weights for the hidden layer")
        new_model.layers[3].set_weights(layer_dict['dense_1'].get_weights())

    elif transfer_mode == "eh":
        print("Setting weights for the embedding layer")
        new_model.layers[1].set_weights(layer_dict['embedding'].get_weights())

        print("Setting weights for the hidden layer")
        new_model.layers[3].set_weights(layer_dict['dense_1'].get_weights())

    return new_model

def ttrain(mode, embedding_matrix, num_words, data, batch_size, epochs, activation_func,
           optimizer, trainable_embd, dropout, weights_path, architecture_path, transfer_mode):

    embedding_dim = len(embedding_matrix[0])

    x_train, y_train, x_dev, y_dev = data[0], data[1], data[2], data[3]
    
    if weights_path and architecture_path:
        weights_filepath = '.'.join(["transfer_weights", mode, str
                                    (embedding_dim), transfer_mode, 
                                    "b", str(batch_size), "e", str(epochs),
                                    activation_func, optimizer,
                                    str(trainable_embd),
                                    str(random.randint(0, 9999)),
                                    str(time.time()), "hdf5"])
    else:
        weights_filepath = '.'.join(["weights_stl", mode, str(embedding_dim), 
                                     "b", str(batch_size), "e", str(epochs),  
                                     activation_func, optimizer, str(trainable_embd),  
                                     str(random.randint(0, 9999)), 
                                     str(time.time()), "hdf5"])
        
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5,
                                   verbose=0, mode='auto') 
    model_checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc',
                                       verbose=1, save_best_only=True,
                                       mode='auto') 
    print("Fine-tuning: %s" % trainable_embd)
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                trainable=trainable_embd,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                name="embedding")

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name="input")
    embedded_sequences = embedding_layer(sequence_input)
    flattened = Flatten()(embedded_sequences)
    dense = Dense(embedding_dim, activation=activation_func, name="dense_1")(flattened)

    if dropout > 0:
        dense = Dropout(dropout)(dense)

    preds = Dense(len(y_train[0]), activation='softmax', name="output")(dense)

    model = Model(sequence_input, preds)

    # model = Sequential()
    # model.add(Embedding(num_words,
    #                     embedding_dim,
    #                     trainable=trainable_embd,
    #                     weights=[embedding_matrix],
    #                     input_length=MAX_SEQUENCE_LENGTH,
    #                     name="embedding"))
    # model.add(Flatten())
    # model.add(Dense(embedding_dim, activation=activation_func, name="dense_1"))

    # if dropout > 0:
    #     model.add(Dropout(dropout))

    # model.add(Dense(len(y_train[0]), activation='softmax', name="output"))

    
    if weights_path and architecture_path:
        model = load_weights(model, weights_path, architecture_path, transfer_mode)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1, # 0 = silent
                        validation_data=(x_dev, y_dev),
                        callbacks=[early_stopping, model_checkpoint])

    model_json = model.to_json()
    with open(mode + "-model.json", "w") as json_file:
        json_file.write(model_json)
    model.summary()
    return model, weights_filepath, history
   
def main():
    argparser = ArgumentParser(description=__doc__)
    argparser.add_argument('-mode', choices=['pcedt', 'nombank'],
                           help='Mode specifies the dataset to use')
    argparser.add_argument('-embedding', help='Path to embedding model (zip file!)')
    argparser.add_argument('-train', help='Path to training split')
    argparser.add_argument('-dev', help='Path to dev split')
    argparser.add_argument('--test', help='Path to test split')
    argparser.add_argument('--weights', help='Which layer weights to transfer')
    argparser.add_argument('--transfer', choices=['e', 'h', 'eh'],
                           help='''Specifies which weights to load in transfer: 
                                   e: embedding,
                                   h: hidden, 
                                   eh: embedding and hidden''')
    argparser.add_argument('--arch', help='''Path to the JSON file of the
                                            auxiliary model architecture in
                                            transfer learning''')
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
    argparser.set_defaults(finetune=True)
    argparser.set_defaults(predict=False)
    
    args = argparser.parse_args(sys.argv[1:])


    if args.mode in ['nombank', 'pcedt']:

        if args.test:
            constituent_index, train_x, train_y, \
                dev_x, dev_y, test_x, test_y, \
                encoder = prepare_traindevtest_imbalanced(args.train, args.dev,
                                                          args.test, args.form)
        else:
            constituent_index, train_x, train_y, \
            dev_x, dev_y, encoder = prepare_traindev_imbalanced(args.train, args.dev, args.form)

        print("Loading the embedding model")
        embedding, dimension = read_embedding(args.embedding)
        embedding_index = prepare_embedding_matrix(embedding, constituent_index,
                                                  dimension)

        data = [train_x, train_y, dev_x, dev_y]

        print("Length of embedding index: %s" % (len(embedding_index)))
        model, weights_filepath, _ = ttrain(args.mode, embedding_index,
                                            len(constituent_index), data,
                                            args.batch, args.epochs,
                                            args.activation, args.optimizer,
                                            args.finetune, args.dropout,
                                            args.weights, args.arch,
                                            args.transfer)

        print("Loading best model weights...")
        best_model = create_old_model(weights_filepath, args.mode + "-model.json")
        best_model.compile(loss='categorical_crossentropy',
                           optimizer=args.optimizer, metrics=['accuracy'])
        print("Evaluation on dev:")
        print(best_model.evaluate(dev_x, dev_y))

        # evaluate on the test set, if given
        if args.test:
            print("---Results on test:")
            print(best_model.evaluate(test_x, test_y))
            true_y = np.argmax(test_y, axis=1)
            predictions = best_model.predict(test_x)
            preds = np.argmax(predictions, axis=1)
            print("Test accuracy: " + str(accuracy_score(true_y, preds)))

        if args.predict:
            if not args.test:
                test_y = dev_y
                test_x = dev_x
            true_y = np.argmax(test_y, axis=1)
            gold = list(encoder.inverse_transform(true_y))
            predictions = best_model.predict(test_x)
            preds = np.argmax(predictions, axis=1)
            pred_classes = list(encoder.inverse_transform(preds))
            
            error_file = open('.'.join(weights_filepath.split('.')[:-1]) + "-errors.txt", 'w')
            #print("F1 score")
            #print(f1_score(true_y, preds, labels=list(set(true_y)), average='weighted'))
            error_file.write("Accuracy\n")
            error_file.write(str(accuracy_score(true_y, preds)) + "\n")
            error_file.write(classification_report(true_y, preds,
                                        labels=list(set(true_y)),
                                        target_names=encoder.inverse_transform(list(set(true_y))),
                                        digits=4))
            error_file.write("\n")
            #print(confusion_matrix(true_y, preds))
            error_file.write("\n")
            inv_map = {v: k for k, v in constituent_index.items()}
            for i in range(len(preds)):
                if preds[i] != true_y[i]:
                    error_file.write('\t'.join([inv_map[test_x[i][0]], 
                                                inv_map[test_x[i][1]], 
                                                str(pred_classes[i]),
                                                str(gold[i]) + "\n"]))
            
            error_file.close()


if __name__ == '__main__':
    main()
