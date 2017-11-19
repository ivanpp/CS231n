import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i,:]
        dW[:,j] += X[i,:]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  num_train = X.shape[0]
  # Get the score matrix
  score = X.dot(W)
  score_mask = np.arange(num_train)
  # Get the correct_label score(reshape for broadcast)
  score_correct = score[score_mask, y].reshape(-1,1)
  # Get the ones matrix
  score_ones = np.ones(score.shape)
  # Refine the ones matrix to ensure the correct label get the margin=0
  score_ones[score_mask, y] = 0
  margin = score - score_correct + score_ones
  margin = np.maximum(0, margin)
  loss += np.sum(margin) / num_train
  loss += reg * np.sum(W * W)  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################+--


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # Get the sign of the margin matrix(no negatives, zero get zero, positive get one)
  margin_sign = np.sign(margin)
  # Since each margin greater than 0 in margin matrix will punish the true class
  # once, we count the number of the positive margin per example, and reverse it 
  # to get the (500,)shape weight array, then use the same way to transfer the 
  # value to the margin_sign matrix.
  label_weight = -np.sum(margin_sign, axis=1)
  margin_sign[score_mask, y] = label_weight
  # Mutiply to get the dW, 0 difference compared with the naive method and 
  # extremely fast.
  dW += (X.T).dot(margin_sign) /num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
