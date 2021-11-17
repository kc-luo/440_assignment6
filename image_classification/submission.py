import collections
import numpy as np


############################################################
# Problem 4.1

def runKMeans(k, patches, maxIter):
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
    p = patches.transpose()
    c = centroids.transpose()
    for i in range(maxIter):
        # BEGIN_YOUR_CODE (around 19 lines of code expected)
        dist = (p[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2
        idx = np.argmin(np.sum(dist, axis=2), axis=1)
        idx_supp = np.transpose(np.eye(k)[idx])
        supp_count = np.sum(idx_supp, axis=1)

        result = p[np.newaxis, :] * idx_supp[:, :, np.newaxis]
        c = np.sum(result.transpose(0, 2, 1), axis=2) / supp_count[:, np.newaxis]
        # END_YOUR_CODE
    return c.transpose()


############################################################
# Problem 4.2

def extractFeatures(patches, centroids):
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
    # features = np.empty((numPatches, k))
    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    p = patches.transpose()
    c = centroids.transpose()
    dist = np.sqrt(((p[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2).sum(axis=2))
    features = np.maximum(np.mean(dist, axis=-1)[:, np.newaxis] - dist, 0)
    # END_YOUR_CODE
    return features


############################################################
# Problem 4.3.1

import math


def logisticGradient(theta, featureVector, y):
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
    e = math.exp(-np.sum(theta*featureVector) * (2*y-1))
    return - e / (1+e) * (2*y-1) * featureVector
    # END_YOUR_CODE


############################################################
# Problem 4.3.2

def hingeLossGradient(theta, featureVector, y):
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
    coeff = (2*y - 1) * featureVector
    if np.sum(theta*coeff) > 1:
        return np.zeros(featureVector.shape)
    else:
        return - coeff
    # END_YOUR_CODE
