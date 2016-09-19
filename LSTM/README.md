### SPARNN
The repository uses some code from the project [SPARNN](https://github.com/sxjscience/SPARNN) which is a Light-Weighted Spatial Temporal RNN Toolbox based on [Theano](http://deeplearning.net/software/theano/install.html)


### Requirements
Latest Python2 (python2.7.*)
numpy + scipy
Theano
HDF5


### Library
4 main components:

`iterator`: data handler

`layer`: network layers, to construct a network, you have to have 3 kinds of layers
```
interface layer: declare input, mask, output
middle layer: construct the main network layers (a list of layers)
cost layer: construct the network cost
```
`model`: network model

`optimizer`: optimizer to optimize the model


### Data Format
The `data_file` is an folder path with a list of hdf5 files for videos:
```
v_ApplyEyeMakeup_g08_c01.h5
v_ApplyEyeMakeup_g08_c02.h5
v_ApplyEyeMakeup_g08_c03.h5
v_ApplyEyeMakeup_g08_c04.h5
v_ApplyEyeMakeup_g08_c05.h5
```
Each hdf5 file stores all the frame features for this video row by row, i.e., a matrix with size (#frames, #featureDim)

The `train_framenum.txt` file contains number of frames for each video:
```
89
123
22
136
```

The `train_filenames.txt` file contains the video filenames relative to the root video directory:
```
v_ApplyEyeMakeup_g08_c01
v_ApplyEyeMakeup_g08_c02
v_ApplyEyeMakeup_g08_c03
v_ApplyEyeMakeup_g08_c04
v_ApplyEyeMakeup_g08_c05
```

The `train_labels.txt`file for single-label datasets looks like
```
0
7
43
```
and for multi-label datasets:
```
0,0,0,0,0,0,0,1,0,0,0,0
0,0,0,0,0,0,0,1,0,0,0,0
0,0,0,0,0,0,1,1,0,0,0,0
0,0,0,0,0,0,0,0,0,0,0,1
```
The same format is required for the validation and test files too.