A demo code that enables to run RGB CNN to generate frame predictions.

Setup
=====

##### Install Caffe
* Build Caffe and pycaffe. The instructions for installing Caffe are [here](http://caffe.berkeleyvision.org/installation.html).

##### Download the datasets and pre-trained models
* Extract all the frames of each video into folders indicated in the annotations files, i.e. **../tvseries/list_test.txt**.

* [Caffe model](http://isis-data.science.uva.nl/zhenyang/online_action/models/tvseries_action_recognition_vgg_16_rgb_iter_3K.caffemodel) trained on tvseries dataset train split, download into **model** folder.


Demo
=======

* Evaluate CNN model on a dataset split
```Shell
python demo_CNN.py
```
(You may have to change the python script according to your own local path)
