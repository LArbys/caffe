import sys, os
import caffe
from ROOT import larcv
larcv.load_pyutil
import numpy as np

class Image2DLayer(caffe.Layer):

    _image2d = None
    _rows = None
    _cols = None

    def setup(self,bottom,top):
        print self.param_str

    def reshape(self,bottom,top):
        if None in [self.__class__._rows, self.__class__._cols]:
            print 'ERROR: rows/cols not provided...'
            raise Exception

        if len(bottom)>0:
            raise Exception('cannot have bottoms for input layer')
        top[0].reshape(1, 1, self.__class__._rows, self.__class__._cols)

    def forward(self, bottom, top):

        if None in [self.__class__._image2d, self.__class__._rows, self.__class__._cols]:
            print 'ERROR: image2d/rows/cols not provided...'
            raise Exception
        
        im = larcv.as_caffe_ndarray(self.__class__._image2d).transpose()
        s  = im.shape
        if not s[0] == self.__class__._rows:
            print 'ERROR: # rows in an image different from config...'
        if not s[1] == self.__class__._cols:
            print 'ERROR: # cols in an image different from config...'

        imm = np.zeros([1, 1, s[0], s[1]],dtype=np.float32)
        imm[0,0,:,:] = im
        top[0].data[...] = imm.astype(np.float32,copy=False)

    def backward(self, top, propagate_down, bottom):
        # no back-prop for input layers
        pass



