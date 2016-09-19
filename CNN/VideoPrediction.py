'''
A sample function for classification using CNN network
Customize as needed:
e.g. num_categories, layer for feature extraction, batch_size
'''

import glob
import os, sys
import numpy as np
import math
import cv2
from scipy.misc import imread, imresize
import scipy.io as sio

caffelib = '/local/softs/caffe'
if caffelib:
    caffepath = caffelib + '/python'
    sys.path.append(caffepath)
import caffe

def VideoPrediction(
        vid_name,
        net,
        num_categories,
        feature_layer,
        start_frame=0,
        num_frames=0,
        num_samples=25
        ):

    if num_frames == 0:
        imglist = glob.glob(os.path.join(vid_name, 'frame_*.jpg'))
        duration = len(imglist)
    else:
        duration = num_frames

    print 'Video: ', vid_name, 'Duration: ', duration, 'Sample: ', num_samples

    # test
    batch_size = 100
    # selection
    step = int(math.floor(duration/num_samples))
    assert step == 1

    prediction = np.zeros((num_categories,num_samples*10), dtype=np.float32)
    N = int(batch_size / 10)

    for i in range(0, num_samples, N):
        in_data = np.zeros((batch_size,3,224,224), dtype=np.float32)

        batch_range = range(i, min(i+N, num_samples))
        batch_rgbs = np.zeros(shape=(len(batch_range)*10,3,224,224), dtype=np.float32)
        for j,k in enumerate(batch_range):
            img_file = os.path.join(vid_name, 'frame_{0:05d}.jpg'.format(k+1))

            #img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img = caffe.io.load_image(img_file)*255.0
            img = img[:,:,(2,1,0)]   #convert RGB-> BGR

            img = cv2.resize(img, (340, 256), interpolation=cv2.INTER_LINEAR)
            img = img - np.array([104., 117., 123.])
            img = np.transpose(img, (2,0,1))
            rgb = img
            rgb_flip = img[:,:,::-1]

            # crop
            batch_rgbs[j*10,:,:,:] = rgb[:, :224, :224]
            batch_rgbs[j*10+1,:,:,:] = rgb[:, :224, -224:]
            batch_rgbs[j*10+2,:,:,:] = rgb[:, 16:240, 60:284]
            batch_rgbs[j*10+3,:,:,:] = rgb[:, -224:, :224]
            batch_rgbs[j*10+4,:,:,:] = rgb[:, -224:, -224:]
            batch_rgbs[j*10+5,:,:,:] = rgb_flip[:, :224, :224]
            batch_rgbs[j*10+6,:,:,:] = rgb_flip[:, :224, -224:]
            batch_rgbs[j*10+7,:,:,:] = rgb_flip[:, 16:240, 60:284]
            batch_rgbs[j*10+8,:,:,:] = rgb_flip[:, -224:, :224]
            batch_rgbs[j*10+9,:,:,:] = rgb_flip[:, -224:, -224:]

        # network forward
        span = range(i*10, min(i+N, num_samples)*10)
        in_data[0:len(batch_range)*10,:,:,:] = batch_rgbs
        out = net.forward(**{net.inputs[0]: in_data})
        output = out[net.outputs[0]]
        prediction[:, span] = np.transpose(output[0:len(batch_range)*10, :])

    return prediction
