import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        diff = X[i] - self.X_train[j]
        square_sum = np.sum(diff * diff)
        dist_l2 = np.sqrt(square_sum)
        dists[i,j] = dist_l2
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      diff = X[i] - self.X_train
      square_sum = np.sum(diff * diff, axis=1)
      dist_l2 = np.sqrt(square_sum)
      dists[i] = dist_l2
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    """
    Method 1:
    Increase the dimension to creat an (500,5000,3072)array and a (500,1,3072)
    one.
    Use broadcast to compute the diff of all train data with all test data,
    and get a MemoryError.
    Cause only one (500,5000,3072) array of int needs (500*5000*3072*28)/(8*2^30)
    GB memory, about 25GB.(The size of int in Python x64 is 28 bits)
    Here is the ez-to-understand code:
    
    # X_train_3d.shape = (500,5000,3072)
    X_train_3d = np.repeat(self.X_train[np.newaxis,:,:], 500, axis=0)
    X_test_3d = X.reshape(500,1,3072)
    # diff.shape = (500,5000,3072)
    diff = X_train_3d - X_test_3d
    diff = diff * diff
    # square_sum.shape = (500,5000)
    square_sum = np.sum(diff, axis=2)
    dists = np.sqrt(square_sum)
    """
    
    """
    Method 2:
    In fact to use broadcast we catually don't need to increase the dimension.
    Notice that the L2 distance formula: sqrt((q-p)^2),which q and p are the 
    same size vectors(In this case 3072*1).
    And (q - p)^2 = q^2 + p^2 - 2 * q * p.
    So we need to calculate 500 q^2, 5000 p^2 and 500*5000 q*p.
    The (500,) size array test_sum gives the 500 q^2 altogether.
    And train_sum gives the 5000 p^2.
    To use the broadcast mechanism in numpy, we should reshape test_sum to (500,1).
    Add the (500,1)array and the (5000,)array give us the (500,5000)array.
    The inner_product gives us 500*5000 q*p.
    So we just add them altogether and do a sqrt to get the l2 distance matrix 'dists'.
    BTW: The no loop method is extremely fast(compared with 2 other methods).
    Code:
    """
    test_sum = np.sum(X * X, axis=1)
    test_sum = test_sum.reshape(-1,1)
    train_sum = np.sum(self.X_train * self.X_train, axis=1)
    inner_product = np.dot(X, self.X_train.T)
    square_sum = test_sum + train_sum - 2 * inner_product
    dists = np.sqrt(square_sum)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      sort = np.argsort(dists[i,:])
      sort = sort[:k]
      closest_y = self.y_train[sort]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      count = np.bincount(closest_y)
      y_pred[i] = np.argmax(count)
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

