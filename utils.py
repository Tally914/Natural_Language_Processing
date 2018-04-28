# from fastai.structured import *
# from fastai.column_data import *
import math, os, json, sys, re
import threading
# import cPickle as pickle  # Python 2
import pickle  # Python3
from IPython.display import HTML
# import xgboost as xg
import coinmarketcap
from coinmarketcap import Market
import requests
import basc_py4chan as py4chan
from datetime import timedelta
from twitter import *
import praw
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
import pandas as pd
from glob import glob
import datetime as dt
import ccxt
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
# import torchtext, torchvision
import keras
import sklearn
from sklearn.utils import shuffle as shfl
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from gensim.models.word2vec import Word2Vec
from pandas.api.types import is_string_dtype, is_numeric_dtype
import numpy as np
from scipy.misc import imresize
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain
import matplotlib.patheffects as PathEffects
import PIL
from sklearn.tree import export_graphviz
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from IPython.lib.deepreload import reload as dreload
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
import bcolz
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import multiprocessing
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import dask.array as da
import multiprocessing
from keras.utils import Sequence
import PIL, os, math, collections, threading, json, bcolz, random, scipy, cv2, io
import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils

from selenium import webdriver

from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Embedding, Reshape, LSTM, Bidirectional, TimeDistributed, Activation, SimpleRNN, GRU, Merge, deserialize, GlobalAveragePooling2D
from keras.layers import SpatialDropout1D
import random, pickle, sys, itertools, string, sys, re, datetime, time, shutil
from operator import itemgetter, attrgetter
from collections import Iterable, Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# from keras_tqdm import TQDMCallback, TQDMNotebookCallback
from keras.layers.core import Flatten, Dense, Dropout, Lambda

# from keras.regularizers import l2, activity_l2, l1, activity_l1  # Keras1
from keras.regularizers import l2, l1  # Keras2

from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from lxml import etree as ET
# from keras.utils.layer_utils import layer_from_config  # Keras1
# from keras.layers import deserialize  # Keras 2
from keras.layers import concatenate
# from keras.layers.merge import dot, add, concatenate  # Keras2
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors, LSHForest
from numpy.random import normal
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import ToktokTokenizer
from functools import reduce
from itertools import chain
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from IPython import get_ipython
# from tqdm import tnrange, tqdm_notebook, tqdm
import xgboost
from ipykernel.kernelapp import IPKernelApp
from IPython.lib.display import FileLink
import IPython
import random,  pickle, sys, itertools, string, sys, re, datetime, time, shutil
from matplotlib import pyplot as plt, rcParams, animation
from ipywidgets import interact, interactive, fixed, widgets, HBox
import matplotlib

from sqlalchemy import create_engine

np.set_printoptions(precision=4, linewidth=90, suppress=True)
matplotlib.rc('animation', html='html5')

client_id = 'lCLLciJOfSC-Sw'
client_secret = '4DAVPdursZHt3ODk6Oo_JFz9fEc'

user_agent = '<platform>:<visbit>:<0.1> (by /u/<Tally914>)'

username = 'Tally914'
password = 'Summit221'

reddit = praw.Reddit(client_id = client_id, client_secret = client_secret, user_agent = user_agent)

auth = OAuth(
	consumer_key='CJAxauDVMXRWimBmoZH8IzSeV',
	consumer_secret='W1HFzPjlQXrrjXagythKIqeKFQmpEi0yYHNCELcK1qLzB7s8gL',
	token='1643070122-W8fxXZxd8axyI0aWD8HCZWDYiUQkw02ZBtfPoGJ',
	token_secret='o0N9e2f2sFntwRcxWx4TEe0kEk0suFQ99oTnbg9i9qVTW')


# def in_notebook(): return IPKernelApp.initialized()
#
# def in_ipynb():
#     try:
#         cls = get_ipython().__class__.__name__
#         return cls == 'ZMQInteractiveShell'
#     except NameError:
#         return False
#
# def clear_tqdm():
#     inst = getattr(tq.tqdm, '_instances', None)
#     if not inst: return
#     for i in range(len(inst)): inst.pop().close()
#
# if in_notebook():
#     def tqdm(*args, **kwargs):
#         clear_tqdm()
#         return tq.tqdm(*args, file=sys.stdout, **kwargs)
#     def trange(*args, **kwargs):
#         clear_tqdm()
#         return tq.trange(*args, file=sys.stdout, **kwargs)
#
# else:
#     from tqdm import tqdm, trange
#     tnrange=trange
#     tqdm_notebook=tqdm
#
#
# def gray(img):
#     if K.image_dim_ordering() == 'tf':
#         return np.rollaxis(img, 0, 1).dot(to_bw)
#     else:
#         return np.rollaxis(img, 0, 3).dot(to_bw)

def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))




def floor(x):
    return int(math.floor(x))
def ceil(x):
    return int(math.ceil(x))

def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def autolabel(plt, fmt='%.2f'):
    rects = plt.patches
    ax = rects[0].axes
    y_bottom, y_top = ax.get_ylim()
    y_height = y_top - y_bottom
    for rect in rects:
        height = rect.get_height()
        if height / y_height > 0.95:
            label_position = height - (y_height * 0.06)
        else:
            label_position = height + (y_height * 0.01)
        txt = ax.text(rect.get_x() + rect.get_width()/2., label_position,
                fmt % height, ha='center', va='bottom')
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])


def column_chart(lbls, vals, val_lbls='%.2f'):
    n = len(lbls)
    p = plt.bar(np.arange(n), vals)
    plt.xticks(np.arange(n), lbls)
    if val_lbls: autolabel(p, val_lbls)

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def do_clip(arr, mx):
    clipped = np.clip(arr, (1-mx)/1, mx)
    return clipped/clipped.sum(axis=1)[:, np.newaxis]


def get_batches(dirname, target_size, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4,  class_mode='categorical'):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def onehot(x):
    return to_categorical(x)


def wrap_config(layer):
    return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}


def copy_layer(layer): return deserialize(wrap_config(layer))  # Keras2


def copy_layers(layers): return [copy_layer(layer) for layer in layers]


def copy_weights(from_layers, to_layers):
    for from_layer,to_layer in zip(from_layers, to_layers):
        to_layer.set_weights(from_layer.get_weights())


def copy_model(m):
    res = Sequential(copy_layers(m.layers))
    copy_weights(m.layers, res.layers)
    return res


def insert_layer(model, new_layer, index):
    res = Sequential()
    for i,layer in enumerate(model.layers):
        if i==index: res.add(new_layer)
        copied = deserialize(wrap_config(layer))  # Keras2
        res.add(copied)
        copied.set_weights(layer.get_weights())
    return res


def adjust_dropout(weights, prev_p, new_p):
    scal = (1-prev_p)/(1-new_p)
    return [o*scal for o in weights]


def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.samples)])  # Keras2


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def xgb_train(trn_data, trn_targets, val_data, val_targets, epochs, xgb_params, plot=False, xgb_model=None):
    x_data = xgboost.DMatrix(trn_data, trn_targets, feature_names=list(trn_data))
    v_data = xgboost.DMatrix(val_data, val_targets, feature_names=list(val_data))
    xg_model = xgboost.train(xgb_params, x_data, evals=[(x_data, 'rmse'), (v_data, 'rmse')], num_boost_round=epochs, verbose_eval=100, xgb_model=xgb_model)
    if plot:
        xgb_plot(xg_model)
    return xg_model


def xgb_plot(model):
    import operator
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')

def save_array(fname, arr, piecemeal=False):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def save_batchwise(fname, batches, num_copies, batch_size, model):
    create=True

    for b in range(len(batches)):
        batch = batches[b]
        num = num_copies[b]
        batch_range = num * int(np.ceil(batch.samples/batch_size))
        count = 1
        for i in range(batch_range):
            if count % 100 == 0:
                print('Predicted batch ' + str(count) + ' out of ' + str(batch_range) + '.')
            conv_feat = model.predict_on_batch(batch.next()[0])
            if create:
                c = bcolz.carray(conv_feat, rootdir=fname, mode='a', chunklen=batch_size)
                count += 1
                create = False
            else:
                c.append(conv_feat)
            count += 1
        print('Computation completed.')
        c.flush()



def load_array(array, batch_size):
    answer = da.from_array(array, chunks=(batch_size,) + array.shape[1:])
    return answer


def mk_size(img, r2c):
    r,c,_ = img.shape
    curr_r2c = r/c
    new_r, new_c = r,c
    if r2c>curr_r2c:
        new_r = floor(c*r2c)
    else:
        new_c = floor(r/r2c)
    arr = np.zeros((new_r, new_c, 3), dtype=np.float32)
    r2=(new_r-r)//2
    c2=(new_c-c)//2
    arr[floor(r2):floor(r2)+r,floor(c2):floor(c2)+c] = img
    return arr


def mk_square(img):
    x,y,_ = img.shape
    maxs = max(img.shape[:2])
    y2=(maxs-y)//2
    x2=(maxs-x)//2
    arr = np.zeros((maxs,maxs,3), dtype=np.float32)
    arr[floor(x2):floor(x2)+x,floor(y2):floor(y2)+y] = img
    return arr

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# def rmse(y_true, y_pred):
#     rmse = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
#     K.update_sub(rmse,  )


def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred / G_true


def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=1)
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
    test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
    return (val_batches.classes, batches.classes, onehot(val_batches.classes), onehot(batches.classes),
            val_batches.filenames, batches.filenames, test_batches.filenames)


def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]

def layer_out_raw(inputs, model, output_name):
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == output_name][0]
    output_layer = model.layers[layer_idx]
    inter_model = Model(inputs=model.inputs, outputs=output_layer.output)
    # inter_model.compile(optimizer=Nadam(), loss=0.001)
    preds = inter_model.predict(inputs, verbose=1)
    pred_list = [preds[index,:] for index in range(len(preds))]

    return pd.DataFrame(pred_list)



class MixIterator(object):
    def __init__(self, iters):
        self.iters = iters
        self.multi = type(iters) is list
        if self.multi:
            self.N = sum([it[0].N for it in self.iters])
        else:
            self.N = sum([it.N for it in self.iters])

    def reset(self):
        for it in self.iters: it.reset()

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        if self.multi:
            nexts = [[next(it) for it in o] for o in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)
        else:
            nexts = [next(it) for it in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)

class BcolzArrayIterator(object):
    """
    Returns an iterator object into Bcolz carray files
    Original version by Thiago Ramon Gonçalves Montoya
    Docs (and discovery) by @MPJansen
    Refactoring, performance improvements, fixes by Jeremy Howard j@fast.ai
        :Example:
        X = bcolz.open('file_path/feature_file.bc', mode='r')
        y = bcolz.open('file_path/label_file.bc', mode='r')
        trn_batches = BcolzArrayIterator(X, y, batch_size=64, shuffle=True)
        model.fit_generator(generator=trn_batches, samples_per_epoch=trn_batches.N, nb_epoch=1)
        :param X: Input features
        :param y: (optional) Input labels
        :param w: (optional) Input feature weights
        :param batch_size: (optional) Batch size, defaults to 32
        :param shuffle: (optional) Shuffle batches, defaults to false
        :param seed: (optional) Provide a seed to shuffle, defaults to a random seed
        :rtype: BcolzArrayIterator

    """

    def __init__(self, X, y=None, w=None, batch_size=32, shuffle=False, seed=None):
        if y is not None and len(X) != len(y):
            raise ValueError('X (features) and y (labels) should have the same length'
                             'Found: X.shape = %s, y.shape = %s' % (X.shape, y.shape))
        if w is not None and len(X) != len(w):
            raise ValueError('X (features) and w (weights) should have the same length'
                             'Found: X.shape = %s, w.shape = %s' % (X.shape, w.shape))
        if batch_size % X.chunklen != 0:
            raise ValueError('batch_size needs to be a multiple of X.chunklen')

        self.chunks_per_batch = batch_size // X.chunklen
        self.X = X
        self.y = y if y is not None else None
        self.w = w if w is not None else None
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed


    def reset(self): self.batch_index = 0


    def next(self):
        with self.lock:
            if self.batch_index == 0:
                if self.seed is not None:
                    np.random.seed(self.seed + self.total_batches_seen)
                self.index_array = (np.random.permutation(self.X.nchunks + 1) if self.shuffle
                    else np.arange(self.X.nchunks + 1))

            #batches_x = np.zeros((self.batch_size,)+self.X.shape[1:])
            batches_x, batches_y, batches_w = [],[],[]
            for i in range(self.chunks_per_batch):
                current_index = self.index_array[self.batch_index]
                if current_index == self.X.nchunks:
                    batches_x.append(self.X.leftover_array[:self.X.leftover_elements])
                    current_batch_size = self.X.leftover_elements
                else:
                    batches_x.append(self.X.chunks[current_index][:])
                    current_batch_size = self.X.chunklen
                self.batch_index += 1
                self.total_batches_seen += 1

                idx = current_index * self.X.chunklen
                if not self.y is None: batches_y.append(self.y[idx: idx + current_batch_size])
                if not self.w is None: batches_w.append(self.w[idx: idx + current_batch_size])
                if self.batch_index >= len(self.index_array):
                    self.batch_index = 0
                    break

            batch_x = np.concatenate(batches_x)
            if self.y is None: return batch_x

            batch_y = np.concatenate(batches_y)
            if self.w is None: return batch_x, batch_y

            batch_w = np.concatenate(batches_w)
            return batch_x, batch_y, batch_w


    def __iter__(self): return self

    def __next__(self, *args, **kwargs): return self.next(*args, **kwargs)


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

def glove_embeddings(glove_location, tokenizer, embedding_dimension):
    word_index = tokenizer.word_index
    embeddings_index = {}
    f = open(glove_location, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = value
    print('Loaded %s word vectors.' % len(embeddings_index))
    f.close()
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector[:embedding_dimension]
    print(f'Glove Embedding matrix of shape {str(embedding_matrix.shape)} generated.')
    del embeddings_index
    return embedding_matrix

def word2vec_init(tokenized_corpus, embedding_dimensions, window_size, iters):
    word2vec = Word2Vec(sentences=tokenized_corpus,
                        size=embedding_dimensions,
                        window=window_size,
                        negative=20,
                        iter=iters,
                        workers=multiprocessing.cpu_count())
    return word2vec


class Seq_Gen(Sequence):
    def __init__(self, x_set, y_set, batch_size, preprocessing=None):
        self.x, self.y = x_set, y_set
        self.batch_size = int(batch_size)
        self.preprocess = preprocessing

    def __len__(self):
        return int(np.ceil(len(self.x) / int(self.batch_size)))

    def __getitem__(self, idx):
        idx = int(idx)
        batch_x = self.x[int(idx * self.batch_size):int((idx + 1) * self.batch_size)]
        batch_y = self.y[int(idx * self.batch_size):int((idx + 1) * self.batch_size)]
        if self.preprocess == None:
            return batch_x, batch_y
        else:
            return self.preprocess(batch_x), batch_y


def trn_val_gens(x_set, y_set, batch_size, val_split, preprocessing=None):
    num_batches = int(np.floor(len(x_set)/batch_size))
    cutoff_batch = int(num_batches * (1 - val_split))
    cutoff = int(cutoff_batch * batch_size)

    x_set, y_set = shfl(x_set, y_set, random_state=42)

    x_train, y_train = x_set[:cutoff], y_set[:cutoff]
    x_valid, y_valid = x_set[cutoff:], y_set[cutoff:]

    assert(len(x_set) == len(y_set))

    train_gen = Seq_Gen(x_train, y_train, batch_size, preprocessing)
    valid_gen = Seq_Gen(x_valid, y_valid, batch_size, preprocessing)

    return train_gen, valid_gen



class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def picklize(location, object):
    with open(location, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Pickle saved to {location}.')

def unpickle(location):
    with open(location, 'rb') as handle:
        return pickle.load(handle)

def split_cols(arr): return np.hsplit(arr, arr.shape[1])

def log_max_inv(preds, max):
    return np.exp(preds * max)

def normalize_inv(preds, ystd, ymean):
    return preds * ystd + ymean

def cat_preproc(dat, cat_map_fit):
    return cat_map_fit.transform(dat).astype(np.int64)

def con_preproc(dat, contin_map_fit):
    return contin_map_fit.transform(dat).astype(np.float32)

def cat_map_info(feat): return feat[0], len(feat[1].classes_)

def struc_emb(feat):
    name, c = cat_map_info(feat)
    c2 = (c+1)//2
    if c2>50: c2=50
    inp = Input((1,), dtype='int64')
    e = Embedding(c, c2, input_length=1)(inp)
    f = Flatten()(e)

    b = BatchNormalization()(f)
    d = Dropout(0.25)(b)
    # s = SpatialDropout1D(0.25)(d)
    # f = Flatten()(s)
    return inp, d


def struc_con(feat, con_dim):
    inp = Input((1,))
    l = Dense(con_dim)(inp)
    b = BatchNormalization()(l)
    d = Dropout(0.25)(b)
    return inp, d

def struc_cons(cons, con_dim):
    inp = Input((len(cons),))
    l = Dense(len(cons) * con_dim)(inp)
    b = BatchNormalization()(l)
    d = Dropout(0.25)(b)
    return inp, d

def reduce_mapped(dataframe, cols):
    df = dataframe.copy()
    df_list = []
    for i in df:
        col_list = []
        for x in i:
            col_list.append(x[0])

        df_list.append(col_list)
    ret = pd.DataFrame(df_list).T
    ret.columns = cols
    return ret


def add_datepart(dataframe, fldname, drop=False):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    """
    df = dataframe
    fld = df[fldname].astype('M8[us]')
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)

    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Hour', 'Minute',
              'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        # def add_col(n): return getattr(fld.dt,n.lower())
        df[targ_pre + "_" + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
    return df





def preprocessing_log(values):
    return plot_array(values, log=True, resize=(224,224), plot=False)

def preprocessing_norm(values):
    return plot_array(values, log=False, resize=(224,224), plot=False)

def plot_array(history_data, log=True, resize=False, plot=False):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    if log == True: ax.loglog(history_data)
    else: ax.plot(history_data)

    ax.axis('off')

    canvas.draw()

    img_single_array = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi()
    img_multi_array = img_single_array.reshape(int(height), int(width), 3)
    if resize != False:
        img_multi_array = imresize(img_multi_array, size=resize)

    if plot == True: plt.imshow(img_multi_array)
    return img_multi_array


def clean_script(raw_data, social_list, social_names, social_lookbacks, group_col, time_col, tar_col, num_pred, num_back):
    df = add_datepart(raw_data, 'Time',  drop=False)

    df_cols = list(df)

    tar_val_cols = [f'{n + 1}_per_tar_val' for n in range(num_pred)]
    back_cols = [f'{n + 1}_per_back' for n in range(num_back)]
    ret_list = []
    sent_cols = []
    for name in social_names:
        for s in social_lookbacks:
            sent_cols.extend((f'{name}_pos_sent_{str(s)}', f'{name}_neg_sent_{str(s)}', f'{name}_net_sent_{str(s)}',
                              f'{name}_count_sent_{str(s)}', f'{name}_prop_pos_sent_{str(s)}',
                              f'{name}_prop_neg_sent_{str(s)}'))

    idx = 0
    for item in list(set(df[group_col].values)):

        g = df[df[group_col] == item]
        g.sort_values(by=[time_col], inplace=True)
        df_list = []
        for i in range(num_back, len(g) - num_pred):
            row = list(g.iloc[i].values)

            all_sents = []
            for i in range(len(social_list)):
                dataframe = social_list[i]
                dt = str_to_dt(g['Time'].iloc[i])
                for s in social_lookbacks:
                    cutoff = dt - timedelta(hours=s)
                    in_time_period = dataframe[dataframe['Timestamp'] > dt_to_int(cutoff)]
                    pos_in = in_time_period['Positive'].astype(float).sum()
                    neg_in = in_time_period['Negative'].astype(float).sum()
                    net_in = in_time_period['Net_Sentiment'].astype(float).sum()
                    count_in = len(in_time_period)
                    prop_pos = pos_in / count_in
                    prop_neg = neg_in / count_in
                    all_sents.extend((pos_in, neg_in, net_in, count_in, prop_pos, prop_neg))

            tar_val_list = [100 * (1 - float(g[tar_col].iloc[i + n + 1]) / float(g[tar_col].iloc[i]))
                            for n in range(num_pred)]

            back_list = [g[tar_col].iloc[i - n - 1] for n in range(num_back)]
            df_list.append([idx] + row + all_sents + tar_val_list + back_list)
            if idx % 10000 == 0:
                print(idx)

            idx += 1

        to_db = pd.DataFrame(df_list)
        ret_list.append(to_db)

    df = pd.concat(ret_list)

    df.columns = ['index'] + df_cols + sent_cols + tar_val_cols + back_cols

    return df

class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape

def dt_to_int(dt): return int(time.mktime(dt.timetuple()))

def int_to_dt(ts): return datetime.datetime.fromtimestamp(ts)

def str_to_dt(string): return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

def get_lookback(val): return datetime.datetime.now() - timedelta(hours=val)


def get_reddit(lookback_list, sub_list, tokenizer, model, client_id=client_id, client_secret=client_secret, user_agent=user_agent):
    max_val = max(lookback_list)
    placeholder = get_lookback(max_val)
    sub_str = "+".join(sub_list)
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    if type(placeholder) == datetime.datetime: placeholder = time.mktime(placeholder.timetuple())
    start = datetime.datetime.fromtimestamp(placeholder)
    end = datetime.datetime.now()
    comment_dict = {}
    subreddit = reddit.subreddit(sub_str)
    for submission in subreddit.submissions(int(time.mktime(start.timetuple())), int(time.mktime(end.timetuple()))):
        if not submission.stickied:
            key = submission.title
            submission.comments.replace_more(limit=10)
            for i, comment in enumerate(submission.comments):
                comment_dict[i] = {
                    'ID': str(comment.id),
                    'Timestamp': comment.created_utc,
                    'Submission': key,
                    'Text': comment.body,
                    'Score': comment.score
                }
    comment_df = pd.DataFrame(comment_dict).T
    unique_comments = comment_df.drop_duplicates(subset='ID', keep='first', inplace=False)
    unique_comment_seqs = tokenizer.texts_to_sequences(unique_comments['Text'].values)
    padded_seqs = pad_sequences(unique_comment_seqs, maxlen=12)
    original_seqs = padded_seqs.shape[0]
    batch_size = model.input_shape[0]
    filler = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    while padded_seqs.shape[0] % batch_size != 0:
        padded_seqs = np.vstack((padded_seqs, filler))
    final_data = np.vstack((padded_seqs, np.zeros(shape=(batch_size * 10, 12))))
    preds = model.predict(final_data, batch_size=128, verbose=0)
    origs = preds[:original_seqs]
    unique_comments['Negative'] = origs[:, 0]
    unique_comments['Positive'] = origs[:, 1]
    unique_comments['Net_Sentiment'] = unique_comments['Positive'] - unique_comments['Negative']

    timeframe_lists = [unique_comments[unique_comments['Timestamp'] >= dt_to_int(get_lookback(ph))] for ph in
                       lookback_list]
    for df in timeframe_lists: print(len(df))
    for lb in lookback_list:
        lb = get_lookback(lb)
        if type(lb) == datetime.datetime: lb = time.mktime(lb.timetuple())
        timing = datetime.datetime.fromtimestamp(lb)
        print(f'Cryptocurrency reddits from {timing} to now.')
    return timeframe_lists


def get_tweets(lookback_list, tokenizer, model, auth=auth):
    t = Twitter(auth=auth)
    max_val = max(lookback_list)
    placeholder = get_lookback(max_val)

    if type(placeholder) == datetime.datetime: placeholder = time.mktime(placeholder.timetuple())
    start = datetime.datetime.fromtimestamp(placeholder)
    full_tweet_dictionary = {}
    timeline = t.statuses.home_timeline(count=1000)

    cryptocurrency = t.search.tweets(q="#cryptocurrency")
    crypto = t.search.tweets(q="#crypto")
    blockchain = t.search.tweets(q="#blockchain")
    altcoin = t.search.tweets(q="#altcoin")

    tweet_list = [timeline, cryptocurrency['statuses'], crypto['statuses'], blockchain['statuses'], altcoin['statuses']]
    for tweets in tweet_list:
        for i, tweet in enumerate(tweets):
            tweet_dictionary = {}
            tweet_dictionary['ID'] = tweet['id_str']
            # tweet_dictionary['Created_At'] = tweet['created_at']
            tweet_dictionary['Timestamp'] = dt_to_int(pd.to_datetime(tweet['created_at']))
            tweet_dictionary['Text'] = tweet['text']
            try:
                tweet_dictionary['Hashtags'] = tweet['entities']['hashtags'][0]['text']
            except:
                tweet_dictionary['Hashtags'] = 'None'
            tweet_dictionary['Text'] = tweet['text']
            tweet_dictionary['Favorites'] = tweet['favorite_count']
            tweet_dictionary['Retweets'] = tweet['retweet_count']
            tweet_dictionary['User_ID'] = tweet['user']['id']
            tweet_dictionary['User_Name'] = tweet['user']['name']
            tweet_dictionary['User_Description'] = tweet['user']['description']
            tweet_dictionary['Follower_Count'] = tweet['user']['followers_count']
            tweet_dictionary['Friend_Count'] = tweet['user']['friends_count']
            full_tweet_dictionary[i] = tweet_dictionary

    tweet_df = pd.DataFrame(full_tweet_dictionary).T
    new_tweets = tweet_df[tweet_df['Timestamp'] >= placeholder]
    unique_comments = new_tweets.drop_duplicates(subset='ID', keep='first', inplace=False)
    unique_comment_seqs = tokenizer.texts_to_sequences(unique_comments['Text'].values)
    padded_seqs = pad_sequences(unique_comment_seqs, maxlen=12)
    original_seqs = padded_seqs.shape[0]
    batch_size = model.input_shape[0]
    filler = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    while padded_seqs.shape[0] % batch_size != 0:
        padded_seqs = np.vstack((padded_seqs, filler))
    final_data = np.vstack((padded_seqs, np.zeros(shape=(batch_size * 10, 12))))
    preds = model.predict(final_data, batch_size=128, verbose=0)
    origs = preds[:original_seqs]
    unique_comments['Negative'] = origs[:, 0]
    unique_comments['Positive'] = origs[:, 1]
    unique_comments['Net_Sentiment'] = unique_comments['Positive'] - unique_comments['Negative']

    timeframe_lists = [unique_comments[unique_comments['Timestamp'] >= dt_to_int(get_lookback(ph))] for ph in
                       lookback_list]
    for lb in lookback_list:
        lb = get_lookback(lb)
        if type(lb) == datetime.datetime: lb = time.mktime(lb.timetuple())
        timing = datetime.datetime.fromtimestamp(lb)
        print(f'Cryptocurrency tweets from {timing} to now.')
    return timeframe_lists


def get_4chan(lookback_list, tokenizer, model):
    biz = py4chan.Board('biz')
    threads = biz.get_all_threads()
    thread_list = []
    post_list = []
    timestamp_list = []
    for thread in threads:
        posts = [post.text_comment for post in thread.replies]
        timestamps = [post.timestamp for post in thread.replies]
        topics = [thread.topic.text_comment for post in thread.replies]
        for post in posts: post_list.append(post.strip('>'))
        for ts in timestamps: timestamp_list.append(ts)
        for topic in topics: thread_list.append(topic)

    post_df = pd.DataFrame(timestamp_list, columns=['Timestamp'])
    post_df['Thread'] = pd.Series(thread_list)
    post_df['Text'] = pd.Series(post_list)

    max_val = max(lookback_list)
    placeholder = get_lookback(max_val)

    if type(placeholder) == datetime.datetime: placeholder = time.mktime(placeholder.timetuple())
    start = datetime.datetime.fromtimestamp(placeholder)

    unique_posts = post_df.drop_duplicates(keep='first', inplace=False)
    unique_comment_seqs = tokenizer.texts_to_sequences(unique_posts['Text'].values)
    padded_seqs = pad_sequences(unique_comment_seqs, maxlen=12)
    original_seqs = padded_seqs.shape[0]
    batch_size = model.input_shape[0]
    filler = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    while padded_seqs.shape[0] % batch_size != 0:
        padded_seqs = np.vstack((padded_seqs, filler))
    final_data = np.vstack((padded_seqs, np.zeros(shape=(batch_size * 10, 12))))
    preds = model.predict(final_data, batch_size=128, verbose=0)
    origs = preds[:original_seqs]
    unique_posts['Negative'] = origs[:, 0]
    unique_posts['Positive'] = origs[:, 1]
    unique_posts['Net_Sentiment'] = unique_posts['Positive'] - unique_posts['Negative']
    timeframe_lists = [unique_posts[unique_posts['Timestamp'] >= dt_to_int(get_lookback(ph))] for ph in lookback_list]
    for lb in lookback_list:
        lb = get_lookback(lb)
        if type(lb) == datetime.datetime: lb = time.mktime(lb.timetuple())
        timing = datetime.datetime.fromtimestamp(lb)
        print(f'Cryptocurrency 4chan posts from {timing} to now.')
    return timeframe_lists

def get_sentiment(df, tokenizer, model):
    unique_comment_seqs = tokenizer.texts_to_sequences(df['Text'].values)
    padded_seqs = pad_sequences(unique_comment_seqs, maxlen=12)
    original_seqs = padded_seqs.shape[0]
    batch_size = model.input_shape[0]
    filler = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    while padded_seqs.shape[0] % batch_size != 0:
        padded_seqs = np.vstack((padded_seqs, filler))
    final_data = np.vstack((padded_seqs, np.zeros(shape=(batch_size * 10, 12))))
    preds = model.predict(final_data, batch_size=128, verbose=0)
    origs = preds[:original_seqs]
    df['Negative'] = origs[:, 0]
    df['Positive'] = origs[:, 1]
    df['Net_Sentiment'] = df['Positive'] - df['Negative']
    return df


def tweet_cleaning(dataset_location, sentiment='Sentiment', text='Text'):
    def clean(tweet):
        if tweet.startswith('"'):
            tweet = tweet[1:]
        if tweet.endswith('"'):
            tweet = tweet[::-1]
        tweet = tweet.strip().lower()
        return tweet
    df = pd.read_csv(dataset_location,  encoding = "cp1252")
    df = df[[sentiment, text]]
    labels = list(df[sentiment].values)
    tweets = list(df[text].map(clean).values)
    corpus = []
    for tweet in tweets:
        corpus.append(' <start> ' + str(tweet) + ' <end> ')
    return corpus, tweets, labels