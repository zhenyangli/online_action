'''
A sample script to run classificition using RGB CNN nets.
Modify this script and path as needed.
'''

import os
import sys
import numpy as np
import math
from os import listdir
from os.path import isfile, join
from scipy import stats

caffelib = '/local/softs/caffe'
if caffelib:
    caffepath = caffelib + '/python'
    sys.path.append(caffepath)
import caffe

from VideoPrediction import VideoPrediction

def softmax(x):
    y = np.exp(x)
    sum_y = y.sum(0)
    z = y / np.tile(sum_y, (y.shape[0],1))
    return z

def main():

    # caffe init
    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    # spatial prediction
    model_def_file = './tvseries_action_rgb_vgg_16_deploy.prototxt'
    model_file = './model/tvseries_action_recognition_vgg_16_rgb_iter_3K.caffemodel'
    spatial_net = caffe.Net(model_def_file, model_file, caffe.TEST)

    # input videos (containing image_*.jpg and some settings)
    dataset = '../tvseries/list_test.txt' # list_val.txt # list_train.txt
    filenames = []
    numframes = []
    with open(dataset) as fp:
        for line in fp:
            splits = line.strip().split(' ')
            filenames.append(splits[0])
            numframes.append(int(splits[1]))

    start_frame = 0
    num_categories = 30
    feature_layer = 'fc8-tvseries'
    save_dir = './result'

    for i, filename in enumerate(filenames):
        filename_ = os.path.splitext(os.path.basename(filename))[0]
        input_video_dir = filename

        # RGB net prediction
        spatial_prediction = VideoPrediction(
                input_video_dir,
                spatial_net,
                num_categories,
                feature_layer,
                start_frame,
                0,
                numframes[i])
        spatial_pred = softmax(spatial_prediction)

        fp_result = open(join(save_dir, filename_ + '.txt'), 'w')
        for j in xrange(numframes[i]):
            # average 10 data augments
            score = spatial_pred[:, j:(j+1)*10]
            avg_score = score.mean(axis=1)
            line = ','.join([str(x) for x in avg_score])
            fp_result.write(line+'\n')
        fp_result.close()

    print '*********** PROCESSED ALL *************'

if __name__ == "__main__":
    main()
