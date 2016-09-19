'''
A sample script to run classificition using both spatial/temporal nets.
Modify this script as needed.
'''
import os
import sys
import numpy as np
import math
from os import listdir
from os.path import isfile, join
from scipy import stats

caffelib = '/home/zhenyang/local/softs/caffe'
if caffelib:
    caffepath = caffelib + '/python'
    sys.path.append(caffepath)
import caffe

from VideoSpatialPrediction_ import VideoSpatialPrediction
from VideoTemporalPrediction import VideoTemporalPrediction

#def softmax(x):
#    y = [math.exp(k) for k in x]
#    sum_y = math.fsum(y)
#    z = [k/sum_y for k in y]
#    return z
def softmax(x):
    y = np.exp(x)
    sum_y = y.sum(0)
    z = y / np.tile(sum_y, (y.shape[0],1))
    return z

def main():

    # caffe init
    gpu_id = 1
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    # spatial prediction
    model_def_file = '../tvseries_action_rgb_vgg_16_deploy.prototxt'
    model_file = '../tvseries_action_recognition_vgg_16_rgb_iter_3K.caffemodel'
    spatial_net = caffe.Net(model_def_file, model_file, caffe.TEST)

    # temporal prediction
    #model_def_file = '../cuhk_action_myflow_vgg_16_deploy.prototxt'
    #model_file = '../lzy_action_recognition_vgg_16_split1_myflow_iter_56K.caffemodel'
    #temporal_net = caffe.Net(model_def_file, model_file, caffe.TEST)

    # input video (containing image_*.jpg and flow_*.jpg) and some settings
    dataset = '../../../examples/tvseries/dataset_file_examples/annot_caffe_test.txt'
    #dataset = '../../../examples/tvseries/dataset_file_examples/annot_caffe_val.txt'
    filenames = []
    numframes = []
    labels = []
    with open(dataset) as fp:
        for line in fp:
            splits = line.strip().split(' ')
            filenames.append(splits[0])
            numframes.append(int(splits[1]))
            labels.append(int(splits[2]))

    start_frame = 0
    num_categories = 30
    feature_layer = 'fc8-tvseries'

    preds = np.zeros((len(filenames),), dtype=np.int64)
    for i, filename in enumerate(filenames):

        #filename_ = os.path.splitext(os.path.basename(filename))[0]
        input_video_dir = filename
        
        # temporal net prediction
        #temporal_mean_file = 'flow_mean.mat'
        #temporal_prediction = VideoTemporalPrediction(
        #        input_video_dir,
        #        temporal_mean_file,
        #        temporal_net,
        #        num_categories,
        #        feature_layer,
        #        start_frame)

        # 1)
        #temporal_pred = softmax(temporal_prediction)
        #temporal_pred = temporal_pred.argmax(axis=0)
        #avg_temporal_pred = stats.mode(temporal_pred)[0][0]
        ## 2)
        #temporal_pred = softmax(temporal_prediction)
        #temporal_pred = temporal_pred.mean(axis=1)
        #avg_temporal_pred = temporal_pred.argmax()

        #preds[i] = int(avg_temporal_pred)

        # spatial net prediction
        spatial_prediction = VideoSpatialPrediction(
                input_video_dir,
                spatial_net,
                num_categories,
                feature_layer,
                start_frame,
                0)
                #numframes[i])

        ## 1)
        ##spatial_pred = softmax(spatial_prediction)
        ##spatial_pred = spatial_pred.argmax(axis=0)
        ##print spatial_pred.shape
        ##avg_spatial_pred = stats.mode(spatial_pred)[0][0]
        ## 2)
        spatial_pred = softmax(spatial_prediction)
        spatial_pred = spatial_pred.mean(axis=1)
        avg_spatial_pred = spatial_pred.argmax()
        
        preds[i] = int(avg_spatial_pred)

        # fused prediction (temporal:spatial = 2:1)
        #fused_pred = np.array(avg_temporal_pred) * 2./3 + \
        #             np.array(avg_spatial_pred) * 1./3
        
        

    # calculate accuracy
    #print preds
    #print preds.argmax(axis=0)
    #print labels
    #preds = preds.argmax(axis=0)
    acc = (preds == np.array(labels)).mean()
    print 'Acc: ', acc

if __name__ == "__main__":
    main()
