import sys, os
from os import listdir
from os.path import isfile, join
import numpy as np
import glob

print np.__file__

if __name__ == "__main__":

    dataset = './annot_caffe_train.txt'
    fp_filenames = open('./train_filenames.txt', 'w')
    fp_framenum = open('./train_framenum.txt', 'w')
    fp_labels = open('./train_labels.txt', 'w')
    with open(dataset) as fp:
        for line in fp:
            splits = line.rstrip().split(' ')
            filename = os.path.splitext(os.path.basename(splits[0]))[0]
            numframe = int(splits[1])
            label = int(splits[2])
            
            fp_filenames.write(filename+'\n')
            fp_framenum.write(str(numframe)+'\n')
            fp_labels.write(str(label)+'\n')

    dataset = './annot_caffe_val.txt'
    fp_filenames = open('./val_filenames.txt', 'w')
    fp_framenum = open('./val_framenum.txt', 'w')
    fp_labels = open('./val_labels.txt', 'w')
    with open(dataset) as fp:
        for line in fp:
            splits = line.rstrip().split(' ')
            filename = os.path.splitext(os.path.basename(splits[0]))[0]
            numframe = int(splits[1])
            label = int(splits[2])
            
            fp_filenames.write(filename+'\n')
            fp_framenum.write(str(numframe)+'\n')
            fp_labels.write(str(label)+'\n')

    dataset = './annot_caffe_test.txt'
    fp_filenames = open('./test_filenames.txt', 'w')
    fp_framenum = open('./test_framenum.txt', 'w')
    fp_labels = open('./test_labels.txt', 'w')
    with open(dataset) as fp:
        for line in fp:
            splits = line.rstrip().split(' ')
            filename = os.path.splitext(os.path.basename(splits[0]))[0]
            numframe = int(splits[1])
            label = int(splits[2])
            
            fp_filenames.write(filename+'\n')
            fp_framenum.write(str(numframe)+'\n')
            fp_labels.write(str(label)+'\n')

    print '*********** PROCESSED ALL *************'

