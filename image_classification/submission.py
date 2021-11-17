import collections
import numpy as np

############################################################
# Problem 4.1

def runKMeans(k,patches,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      patches - 2D numpy array of size patchSize x numPatches
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size patchSize x k
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    centroids = np.random.randn(patches.shape[0], k)
    numPatches = patches.shape[1]

    for i in range(maxIter):
        # BEGIN_YOUR_CODE (around 19 lines of code expected)
        p_expanded = np.repeat(patches[:, np.newaxis, :], k, axis=1)
        c_expanded = np.repeat(centroids[:, :, np.newaxis], numPatches, axis=2)
        dist = (p_expanded - c_expanded) ** 2
        idx = np.argmin(dist, axis=1)
        idx = np.transpose(idx)
        idx_supp = np.eye(k)[idx]
        print(idx_supp.shape)

        supp_count = np.sum(idx_supp, axis=1)
        print(supp_count)
        result = np.repeat(patches[np.newaxis, :], k, axis=0) * idx_supp[:, :, np.newaxis]

        ################################################################################
        #             END OF YOUR CODE                                                 #
        ################################################################################
        #     return centroids
        centroids = np.sum(result.transpose(0, 2, 1), axis=2) / supp_count[:, np.newaxis]
        # END_YOUR_CODE

    return centroids

############################################################
# Problem 4.2

def extractFeatures(patches,centroids):
    """
    Given patches for an image and a set of centroids, extracts and return
    the features for that image.
    
    Args:
      patches - 2D numpy array of size patchSize x numPatches
      centroids - 2D numpy array of size patchSize x k
      
    Returns:
      features - 2D numpy array with new feature values for each patch
                 of the image in rows, size is numPatches x k
    """
    k = centroids.shape[1]
    numPatches = patches.shape[1]
    features = np.empty((numPatches,k))

    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    raise Exception("Not yet implemented")
    # END_YOUR_CODE
    return features

############################################################
# Problem 4.3.1

import math
def logisticGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of the logistic loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of logistic loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    raise Exception("Not yet implemented.")
    # END_YOUR_CODE

############################################################
# Problem 4.3.2
    
def hingeLossGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of hinge loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of hinge loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    raise Exception("Not yet implemented.")
    # END_YOUR_CODE

