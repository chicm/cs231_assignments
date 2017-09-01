import numpy as np
from random import shuffle
from past.builtins import xrange

def stablesoftmax(x):
  """Compute the softmax of vector x in a numerically stable way."""
  shiftx = x - np.max(x)
  exps = np.exp(shiftx)
  return exps / np.sum(exps)

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  #yp = np.zeros((X.shape[0], W.shape[1]))
  N = X.shape[0]
  C = W.shape[1]
  regvalue = np.sum(np.square(W)) * reg

  scores = np.dot(X,W)
  probs = np.zeros_like(scores)
  dscores = np.zeros_like(scores)
 
  for n in range(N):
    yp = stablesoftmax(scores[n])
    lossi = -np.log(yp[y[n]]) 
    loss += lossi 
    for i in range(C):
        indicator = 1.0 if i == y[n] else 0
        dscores[n,i] += yp[i] - indicator
    
  loss /= N
  loss += regvalue
  dW = np.dot(X.T, dscores)
  dW  /= N
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

