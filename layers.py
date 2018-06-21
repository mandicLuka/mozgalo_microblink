from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #Find shape of X
    shapes = np.array(x.shape)
    shapes[0] = 1
    d_shape = np.prod(shapes)

    X = x.reshape(x.shape[0],d_shape)
    out = np.dot(X,w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    dx, dw, db = None, None, None


    x, w, b = cache

    shape = np.array(x.shape)
    shape[0] = 1
    d_shape = np.prod(shape)
    shape[0] = x.shape[0]


    X = x.reshape(x.shape[0],d_shape)


    dx = np.dot(dout,w.T)
    dw = np.dot(X.T,dout)
    db = np.sum(dout,axis=0)

    dx = np.reshape(dx,x.shape)
    x, w, b = cache

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = x.copy()

    

    out[x < 0] = 0
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = np.ones(x.shape)
    dx[x <= 0] = 0
    dx = dout * dx
    return dx


def spatial_pooling_forward(x, shape):

    '''Input:
    - x: Input data of shape (N, H, W)
    -pool_list
    '''

    H, W = x.shape

    row_length = np.float(H) / shape[0] 
    col_length = np.float(W) / shape[1] 

    outputs = []
    
    for i in range(shape[0]):

        x1 = i * row_length
        x2 = i * row_length + row_length
        x1 = np.int(np.ceil(x1))
        x2 = np.int(np.ceil(x2))
        if x1 == x2:
            if shape[0] - i < 5:
                x1 = x2 - np.int(np.ceil(row_length))
            else:
                x2 = x1 + np.int(np.ceil(row_length))
            
        for j in range(shape[1]):

            y1 = j * col_length
            y2 = j * col_length + col_length
            y1 = np.int(np.ceil(y1))
            y2 = np.int(np.ceil(y2))
            if y1 == y2:
                if shape[1] - j < 5:
                    y1 = y2 - np.int(np.ceil(col_length))
                else:
                    y2 = y1 + np.int(np.ceil(col_length))
            x_crop = x[x1:x2, y1:y2]
            
            pooled_val = np.mean(x_crop)
            if not x_crop.size:
                pooled_val = x[min(x1, H), min(y1, W)]
            outputs.append(np.int(pooled_val))

    outputs = np.array(outputs)
    outputs = outputs.reshape(shape)
    return outputs



def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)

    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)

    N = x.shape[0]

    loss = -1 * np.sum(log_probs[np.arange(N), np.argmax(y, axis=1)]) / N
    dx = probs.copy()
    dx[np.arange(N), np.argmax(y, axis=1)] -= 1
    dx /= N
    return loss, dx
