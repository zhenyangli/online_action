A DEMO code that enables to run LSTM to predict frame scores.

Setup
=====

##### Install Theano and other python packages
* Install Theano. The instructions for installing Theano are [here](http://deeplearning.net/software/theano/install.html).
* Required python packages:
    numpy + scipy
    HDF5

##### Download the datasets and pre-trained models
* Extract all the frames of each video into folders indicated in the annotations files, i.e. **../tvseries/list_test.txt**.

* [LSTM model](http://caffe.berkeleyvision.org/installation.html) trained on tvseries dataset train split, download into **model** folder.


Demo
=======

* Extact per-frame CNN features
```Shell
python extract_features/extract_rgbcnn.py
```

* Evaluate LSTM model on a dataset split
```Shell
python demo_LSTM.py
```
(You may have to change the python script according to your own local path)
