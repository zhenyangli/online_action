__author__ = 'zhenyang'

import theano
import theano.tensor as TT

import sparnn
import sparnn.utils
from sparnn.utils import *

from sparnn.iterators import AdvancedVideoIterator
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
import h5py


label_dims = [30]
def get_acc(model, data_iterator, save_dir):

    probs = numpy.zeros((data_iterator.total(),data_iterator.seq_length)+tuple(label_dims)).astype(theano.config.floatX)

    old_mode = model.mode
    model.set_mode('predict')
    data_iterator.begin(do_shuffle=False)
    while True:
        batch_data = data_iterator.get_batch()
        prob = model.output_func_dict['probability'](*(batch_data)) # (TS, BS, #actions)
        prob = numpy.transpose(prob, (1,0,2))
        probs[data_iterator.current_batch_indices, :, :] = prob[:data_iterator.current_batch_size, :, :]

        data_iterator.next()
        if data_iterator.no_batch_left():
            break
    model.set_mode(old_mode)

    for vid_idx, video_name in enumerate(data_iterator.video_names):
        vid_duration = data_iterator.num_frames[vid_idx]
        print video_name, vid_duration
        vid_prob = numpy.zeros((vid_duration,)+tuple(label_dims)).astype(theano.config.floatX)
        vid_count = numpy.zeros((vid_duration,)).astype(theano.config.floatX)

        ##
        segments = numpy.where(data_iterator.video_indices == vid_idx)[0]
        print len(segments)
        for seg_idx in segments:
            start = data_iterator.frame_local_indices[seg_idx]
            end = start + data_iterator.seq_length * data_iterator.seq_skip

            if end > vid_duration:
                assert start == 0
                assert segments.size == 1
                vid_prob[...] = probs[seg_idx, :vid_duration, :]
                vid_count = vid_count + 1.
            else:
                if start == 0:
                    vid_prob[start:end, :] = vid_prob[start:end, :] + probs[seg_idx, :, :]
                    vid_count[start:end] = vid_count[start:end] + 1.
                else:
                    vid_prob[end-1, :] = vid_prob[end-1, :] + probs[seg_idx, -1, :]
                    vid_count[end-1] = vid_count[end-1] + 1.

        print numpy.amax(vid_count)
        # average
        assert numpy.all(vid_count>0)
        assert numpy.all(vid_count<2)
        vid_prob = vid_prob / vid_count[:, numpy.newaxis]

        # save results
        with open(os.path.join(save_dir, video_name+'.txt'), "w") as fp:
            for j in xrange(vid_duration):
                score = vid_prob[j, :]
                line = ','.join([str(x) for x in score])
                fp.write(line+'\n')


save_path = "./results/"
model_path = './models/tvseries-baseline-lstm-fc6-seq-16/tvseries-VideoModel-LSTM-RMS-validation-best.pkl'
log_path = save_path + "test_tvseries_lstm.log"
result_save_dir = save_path + 'results-stride-1-lastpred'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)

sparnn.utils.quick_logging_config(log_path)

random.seed(1000)
numpy.random.seed(1000)

seq_length = 16


#############################
iterator_param = {'dataset': 'tvseries',
                  'data_file': '/data/tvseries/features/rgb_vgg16_fc6',
                  'num_frames_file': '/data/tvseries/test_full_framenum.txt',
                  'labels_file': '/data/tvseries/test_full_labels.txt',
                  'vid_name_file': '/data/tvseries/test_full_filenames.txt',
                  'dataset_name': 'features', 'rng': None,
                  'seq_length': seq_length, 'seq_stride': 1, 'seq_fps': 30,
                  'minibatch_size': 256,
                  'use_mask': True, 'input_data_type': 'float32', 'output_data_type': 'int64', 'one_hot_label': True,
                  'is_output_multilabel': False,
                  'name': 'tvseries-valid-advanced-video-iterator'}
valid_iterator = AdvancedVideoIterator(iterator_param)
valid_iterator.begin(do_shuffle=False)
valid_iterator.print_stat()


############################# model
model = VideoModel.load(model_path)
model.print_stat()


############################# optimizer
accuracy = quick_timed_log_eval(logger.info, "Validation Accuracy", get_acc,
                                *(model, valid_iterator, result_save_dir))
print "Validation Accuracy: ", str(accuracy)

