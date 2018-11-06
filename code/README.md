This folder contains the Python scripts and data splits used to run the
experiments presented in 
__Transfer and Multi-Task Learning for Noun-Noun Compound Interpretation__
by [Fares et al. (2018)](https://arxiv.org/abs/1809.06748) in _Proceedings of
the 2018 Conference on Empirical Methods in Natural Language Processing 
(EMNLP)_


### [transfer.py](transfer.py)

This script can be used to train both single-task learning (STL) models and
transfer learning (TL) models.

STL training example: 
```console 
python transfer.py -mode nombank -train data/nombank_train.conll \\
-dev data/nombank_dev.conll --test data/nombank_test.conll \\ 
-embedding [path  to embedding model] --verbose
```

Transfer learning example: 
```console 
python transfer.py -mode nombank -train data/nombank_train.conll \\
-dev data/nombank_dev.conll --test data/nombank_test.conll \\ 
-embedding [path to embedding model] --verbose \\
--transfer e --arch pcedt-model.json --weights [path to the aux model weights]
```

Arguments:
```
  -h, --help            show this help message and exit
  -mode {pcedt,nombank}
                        Mode specifies the dataset to use
  -embedding EMBEDDING  Path to embedding model (zip file!)
  -train TRAIN          Path to training split
  -dev DEV              Path to dev split
  --test TEST           Path to test split
  --weights WEIGHTS     hdf5 file of the auxiliary model
  --transfer {e,h,eh}   Specifies which weights to load in transfer: e:
                        embedding, h: hidden, eh: embedding and hidden
  --arch ARCH           Path to the JSON file of the auxiliary model
                        architecture in transfer learning
  --nofinetune          Do not update the embedding layer.
  --batch BATCH         Batch size. Default: 5
  --epochs EPOCHS       # of epochs. Default: 100
  --optimizer {RMSprop,adam,adamax,adadelta,SGD,Adagrad}
                        Default: adam
  --activation {sigmoid,relu}
                        Activation function of the hidden layer. Default:
                        sigmoid
  --dropout DROPOUT     Dropout rate. Default: 0
  --form {fullform,lemma}
                        Token form in compound dataset. Default: fullform
  --verbose             Write detailed classification report and
                        classification errors to a file
```


### [multitask.py](multitask.py)

This script is used to train multi-task learning (MTL) models. 

Example use:

```console
python multitask.py -mode nomfun -mtlmode mtle -embedding [path to embedding model] --verbose
```
Arguments:
```
  -h, --help            show this help message and exit
  -mode {nomfun,funnom}
                        Mode specifies the main and auxiliary tasks. nomfun:
                        NomBank is main and PCEDT is auxiliary. funnom: PCEDT
                        is main and NomBank is auxiliary
  -mtlmode {mtle,mtlf}  Specifies MTL setup: mtle (default) shares the
                        embedding layer only. mtlf shares the embedding and
                        hidden layers.
  -embedding EMBEDDING  Path to embedding model (zip file!)
  --test                Evaluate on the test split.
  --nofinetune          Do not update the embedding layer.
  --batch BATCH         Batch size. Default: 5
  --epochs EPOCHS       # of epochs. Default: 100
  --optimizer {RMSprop,adam,adamax,adadelta,SGD,Adagrad}
                        Default: adam
  --activation {sigmoid,relu}
                        Activation function of the hidden layer. Default:
                        sigmoid
  --dropout DROPOUT     Dropout rate. Default: 0
  --form {fullform,lemma}
                        Token form in compound dataset. Default: fullform
  --verbose             Write detailed classification report and
                        classification errors to a file
```


#### Requirements

The experiments in [Fares et al. (2018)](https://arxiv.org/abs/1809.06748) were
run using Python 2.7, Tensorflow  (1.8.0), Keras (2.2.0) and
numpy (1.13.3). All the aforementioned packages are required to run the script
in addition to scikit_learn.
