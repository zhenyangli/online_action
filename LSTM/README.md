A demo code that enables to run LSTM to predict frame scores.

Setup
=====

##### Install Theano and other python packages
* Install Theano. The instructions for installing Theano are [here](http://deeplearning.net/software/theano/install.html).
* Required python packages: numpy + scipy + HDF5

##### Download the datasets and pre-trained models
* Extract all the frames of each video into folders indicated in the annotations files, i.e. **../tvseries/list_test.txt**.

* [Caffe model](http://caffe.berkeleyvision.org/installation.html) trained on tvseries dataset train split, download into **model** folder.
* [LSTM model](http://caffe.berkeleyvision.org/installation.html) trained on tvseries dataset train split, download into **model** folder.


Demo
=======

* Extact per-frame CNN features using Caffe model
```Shell
python extract_features/extract_rgbcnn.py -d ../tvseries/list_test.txt 
--model_def tvseries_action_rgb_vgg_16_deploy_features_fc6.prototxt
--model model/tvseries_action_recognition_vgg_16_rgb_iter_3K.caffemodel
```

* Evaluate LSTM model on a dataset split
```Shell
python demo_LSTM.py
```
(You may have to change the python script according to your own local path)
