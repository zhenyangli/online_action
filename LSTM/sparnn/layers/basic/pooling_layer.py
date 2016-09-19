
import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer

logger = logging.getLogger(__name__)


class PoolingLayer(Layer):
    def __init__(self, layer_param):
        super(PoolingLayer,self).__init__(layer_param)

    def set_name(self):
        self.name = "PoolingLayer-" + str(self.id)