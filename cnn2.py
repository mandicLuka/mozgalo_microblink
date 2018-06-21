from builtins import object
import numpy as np

from layers.layers import *
from layers.fast_layers import *
from layers.layer_utils import *
import os
import pickle

class ThreeLayerConvNet(object):
    """

    TODO experimentations with different layers (add one more 5x5 layer and 
         one more fc layer )

    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_sizes=(7, 3),
                 hidden_dims=(100, 100), num_classes=10, weight_scale=1e-3, dropout=0.5,
                 dtype=np.float64):
        
        
        
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = 0
        self.dropout_param = {'p' : dropout, 'mode' : 'train'}
        self.dtype = dtype
        C,H,W = input_dim

        self.params['W1'] = np.random.normal(0, weight_scale,[num_filters, C, filter_sizes[0], filter_sizes[0]])
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(0, weight_scale,[num_filters, num_filters, filter_sizes[1], filter_sizes[1]])
        self.params['b2'] = np.zeros(num_filters)
        self.params['W3'] = np.random.normal(0, weight_scale,[np.int(H/2) * np.int(W/2) * num_filters, hidden_dims[0]])
        self.params['b3'] = np.zeros(hidden_dims[0])
        self.params['W4'] = np.random.normal(0, weight_scale,[hidden_dims[0], hidden_dims[1]])
        self.params['b4'] = np.zeros(hidden_dims[1])
        self.params['W5'] = np.random.normal(0, weight_scale,[hidden_dims[1], num_classes])
        self.params['b5'] = np.zeros(num_classes)


        self.conv_param_1 = {'stride': 1, 'pad': (filter_sizes[0] - 1) // 2}
        self.conv_param_2 = {'stride': 1, 'pad': (filter_sizes[1] - 1) // 2}
        self.pool_param_1 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        self.pool_param_2 = {'pool_height': 1, 'pool_width': 1, 'stride': 1}

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        

        # pass pool_param to the forward pass for the max-pooling layer
        
        if y is None:
            self.dropout_param['mode'] = 'test'

        scores = None


        pool_out_1, conv_relu_pool_cache_1 = conv_relu_pool_forward(X, W1, b1, self.conv_param_1, self.pool_param_1)

        pool_out_2, conv_relu_pool_cache_2 = conv_relu_pool_forward(pool_out_1, W2, b2, self.conv_param_2, self.pool_param_2)

        dropout_out_1, dropout_cache_1 = dropout_forward(pool_out_2, self.dropout_param)

        relu_out_1, affine_relu_cache_1 = affine_relu_forward(dropout_out_1, W3, b3)

        dropout_out_2, dropout_cache_2 = dropout_forward(relu_out_1, self.dropout_param)

        relu_out_2, affine_relu_cache_2 = affine_relu_forward(relu_out_1, W4, b4)

        affine_out, affine_cache = affine_forward(relu_out_2, W5, b5)

        scores = np.copy(affine_out)

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dSoftmax = softmax_loss(scores, y)
        loss += self.reg * 0.5 * (np.sum(np.square(W1)) + np.sum(np.square(W3)) + np.sum(np.square(W4) + np.sum(np.square(W5))))

        dx5, dw5, db5 = affine_backward(dSoftmax, affine_cache)

        dx4, dw4, db4 = affine_relu_backward(dx5, affine_relu_cache_2)

        ddropout_2 = dropout_backward(dx4, dropout_cache_2)

        dx3, dw3, db3 = affine_relu_backward(ddropout_2, affine_relu_cache_1)

        ddropout_1 = dropout_backward(dx3, dropout_cache_1)

        dx2, dw2, db2 = conv_relu_pool_backward(ddropout_1, conv_relu_pool_cache_2)

        dx1, dw1, db1 = conv_relu_pool_backward(dx2, conv_relu_pool_cache_1)

        grads['W1'], grads['b1'] = dw1 + self.reg * W1, db1
        grads['W2'], grads['b2'] = dw2 + self.reg * W2, db2
        grads['W3'], grads['b3'] = dw3 + self.reg * W3, db3
        grads['W4'], grads['b4'] = dw4 + self.reg * W4, db4
        grads['W5'], grads['b5'] = dw5 + self.reg * W5, db5


        return loss, grads


    def load_params(self, filename):
        current_dir = os.getcwd()
        f = open(current_dir + "/" + filename)
        params = pickle.load(f)
        self.params = params["model"].params
