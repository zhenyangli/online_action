__author__ = 'zhenyang'

import theano
import theano.tensor as TT

import sparnn
import sparnn.utils
from sparnn.utils import *

from sparnn.iterators import VideoDataIterator
from sparnn.layers import InterfaceLayer
from sparnn.layers import FeedForwardLayer
from sparnn.layers import LSTMLayer
from sparnn.layers import DropoutLayer
from sparnn.layers import PredictionLayer
from sparnn.layers import ElementwiseCostLayer

from sparnn.models import VideoModel

from sparnn.optimizers import SGD
from sparnn.optimizers import RMSProp
from sparnn.optimizers import AdaDelta
from sparnn.optimizers import Adam

import os
import random
import numpy


#save_path = "./tvseries-experiment/tvseries-baseline-lstm-fc6-seq-16/rms-lr-0.001-decayr-0-decays-0-drop-0.5-0.7-l2-1e5-init-gauss0.1/"
save_path = "./tvseries-experiment/tvseries-baseline-lstm-fc6-seq-16/rms-lr-0.001-decayr-0-decays-0-drop-0.5-0.5-l2-1e5-init-gauss0.1/"
log_path = save_path + "evaluate_tvseries_lstm.log"

if not os.path.exists(save_path):
    os.makedirs(save_path)

sparnn.utils.quick_logging_config(log_path)

random.seed(1000)
numpy.random.seed(1000)

iterator_rng = sparnn.utils.quick_npy_rng(1337)
iterator_frame_rng = sparnn.utils.quick_npy_rng(1234)
seq_length = 16

#############################
# NOTE: the real batch_size = minibatch_size (number of videos) * num_segments (per video)
iterator_param = {'dataset': 'tvseries',
                  'data_file': '/home/zhenyang/Workspace/data/tvseries/features_dissected/rgb_vgg16_fc6',
                  'num_frames_file': '/home/zhenyang/Workspace/data/tvseries/train_framenum.txt',
                  'labels_file': '/home/zhenyang/Workspace/data/tvseries/train_labels.txt',
                  'vid_name_file': '/home/zhenyang/Workspace/data/tvseries/train_filenames.txt',
                  'dataset_name': 'features', 'rng': iterator_rng, 'frame_rng': iterator_frame_rng,
                  'seq_length': seq_length, 'num_segments': 1, 'seq_fps': 30,
                  'minibatch_size': 64, 'train_sampling': True,
                  'use_mask': True, 'input_data_type': 'float32', 'output_data_type': 'int64', 'one_hot_label': True,
                  'is_output_multilabel': False,
                  'name': 'tvseries-train-video-iterator'}
train_iterator = VideoDataIterator(iterator_param)
train_iterator.begin(do_shuffle=True)
train_iterator.print_stat()
##
iterator_param = {'dataset': 'tvseries',
                  'data_file': '/home/zhenyang/Workspace/data/tvseries/features_dissected/rgb_vgg16_fc6',
                  'num_frames_file': '/home/zhenyang/Workspace/data/tvseries/test_framenum.txt',
                  'labels_file': '/home/zhenyang/Workspace/data/tvseries/test_labels.txt',
                  'vid_name_file': '/home/zhenyang/Workspace/data/tvseries/test_filenames.txt',
                  'dataset_name': 'features', 'rng': None, 'frame_rng': None,
                  'seq_length': seq_length, 'num_segments': 25, 'seq_fps': 30,
                  'minibatch_size': 20, 'train_sampling': False,
                  'use_mask': True, 'input_data_type': 'float32', 'output_data_type': 'int64', 'one_hot_label': True,
                  'is_output_multilabel': False,
                  'name': 'tvseries-valid-video-iterator'}
valid_iterator = VideoDataIterator(iterator_param)
valid_iterator.begin(do_shuffle=False)
valid_iterator.print_stat()
##
test_iterator = None

#############################
rng = sparnn.utils.quick_npy_rng()
theano_rng = sparnn.utils.quick_theano_rng(rng)

############################# interface layer
param = {"id": "tvseries-vgg16-fc6", "use_mask": True,
         "input_ndim": 3, "output_ndim": 2,
         "output_data_type": "int64"}
interface_layer = InterfaceLayer(param)

x = interface_layer.input
mask = interface_layer.mask
y = interface_layer.output

timesteps = x.shape[0]
minibatch_size = x.shape[1]

feature_dim = 4096
hidden_dim = 512
out_dim = 1024
actions = 30
data_dim = (feature_dim,)

logger.info("Data Dim:" + str(data_dim))


############################# middle layers
middle_layers = []

#0# lstm layer (main layer)
param = {"id": 0, "rng": rng, "theano_rng": theano_rng,
         "dim_in": data_dim, "dim_out": (hidden_dim,),
         "minibatch_size": minibatch_size,
         "input": x, "mask": mask,
         #"init_hidden_state": ,
         #"init_cell_state": ,
         "n_steps": seq_length}
middle_layers.append(LSTMLayer(param))

#0.0# build multi-layer lstm if you want
#param = {"id": 0, "rng": rng, "theano_rng": theano_rng,
#         "dim_in": data_dim, "dim_out": (hidden_dim,),
#         "minibatch_size": minibatch_size,
#         "input": middle_layers[0].output, "mask": mask,
#         #"init_hidden_state": ,
#         #"init_cell_state": ,
#         "n_steps": seq_length}
#middle_layers.append(LSTMLayer(param))

#1# set up dropout 1
param = {"id": 1, "rng": rng, "theano_rng": theano_rng,
          "dim_in": (hidden_dim,), "dim_out": (hidden_dim,),
          "minibatch_size": minibatch_size,
          "dropout_rate": 0.5,
          "input": middle_layers[0].output}
middle_layers.append(DropoutLayer(param))

#2# output fc layer
param = {"id": 2, "rng": rng, "theano_rng": theano_rng,
          "dim_in": (hidden_dim,), "dim_out": (out_dim,),
          "minibatch_size": minibatch_size,
          "activation": "tanh",
          "input": middle_layers[1].output}
middle_layers.append(FeedForwardLayer(param))

#3# set up dropout 2
param = {"id": 3, "rng": rng, "theano_rng": theano_rng,
          "dim_in": (out_dim,), "dim_out": (out_dim,),
          "minibatch_size": minibatch_size,
          "dropout_rate": 0.5,
          "input": middle_layers[2].output}
middle_layers.append(DropoutLayer(param))

#4# classification layer (softmax outputs class probabilities)
param = {"id": 4, "rng": rng, "theano_rng": theano_rng,
          "dim_in": (out_dim,), "dim_out": (actions,),
          "minibatch_size": minibatch_size,
          "activation": "softmax",
          "input": middle_layers[3].output}
middle_layers.append(FeedForwardLayer(param))

#5# label prediction layer
param = {"id": 5, "rng": rng, "theano_rng": theano_rng,
         "dim_in": (actions,), "dim_out": (1,),
         "minibatch_size": minibatch_size,
         "last_n": seq_length,
         "is_multilabel": False,
         "input": middle_layers[4].output}
middle_layers.append(PredictionLayer(param))

############################# cost layer
param = {"id": "cost", "rng": rng, "theano_rng": theano_rng,
         "dim_in": (actions,), "dim_out": (1,),
         "minibatch_size": minibatch_size,
         "cost_func": "CategoricalCrossEntropy",
         "regularization": "l2",
         "param_layers": middle_layers,
         "penalty_rate": 0.00001,
         "input": middle_layers[4].output,
         "mask": mask,
         "target": y}
cost_layer = ElementwiseCostLayer(param)

outputs = [{"name": "probability", "value": middle_layers[4].output}, {"name": "prediction", "value": middle_layers[5].output}]


############################# model
param = {'interface_layer': interface_layer, 'middle_layers': middle_layers, 'cost_layer': cost_layer,
         'outputs': outputs, 'errors': None, 'last_n': middle_layers[5].last_n,
         'name': "tvseries-VideoModel-LSTM-RMS",
         'problem_type': "classification"}
model = VideoModel(param)
model.print_stat()


############################# optimizer
param = {'id': '1', 'learning_rate': 0.001, 'momentum': 0.9, 'decay_rate': 0.9, 'clip_threshold': None, 'verbose': False,
         'max_epoch': 400, 'start_epoch': 0, 'valid_epoch': 10, 'max_epochs_no_best': 400, 'decay_step': 400,
         'display_freq': 100, 'valid_freq': None, 'save_freq': None,
         'autosave_mode': ['interval', 'best'], 'save_path': save_path, 'save_interval': 10}
optimizer = RMSProp(model, train_iterator, valid_iterator, test_iterator, param)

optimizer.train()

