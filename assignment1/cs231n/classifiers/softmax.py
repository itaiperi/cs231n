import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)
    scores_exp = np.exp(scores)
    scores_exp_sum = np.sum(scores_exp)
    probs = scores_exp / scores_exp_sum
    loss += -np.log(probs[y[i]])
    for j in range(W.shape[1]):
      if y[i] == j:
        # dW[:, j] = -1. / probs[y[i]] * (X[i] * probs[y[i]] * (1 - probs[y[i]]))
        #          = X[i] * (probs[y[i]] - 1)
        dW[:, j] += X[i] * (probs[y[i]] - 1)
      else:
        # dW[:, j] = -1. / probs[y[i]] * (-X[i] * probs[j] * probs[y[i]])
        #          = X[i] * probs[j]
        dW[:, j] += X[i] * probs[j]

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  scores = X.dot(W)
  scores_exp = np.exp(scores)
  scores_exp_sum = np.sum(scores_exp, axis=1).reshape(-1, 1)
  probs = scores_exp / scores_exp_sum
  loss = np.mean(-np.log(probs[range(y.shape[0]), y]))

  # -1 for the right labels, because of the equation of the derivative.
  probs[range(y.shape[0]), y] -= 1
  dW = X.T.dot(probs)
  dW /= X.shape[0]

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

