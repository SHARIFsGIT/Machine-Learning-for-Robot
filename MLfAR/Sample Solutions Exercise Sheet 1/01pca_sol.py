import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

def pca_with_svd(data):
    '''
    TODO
    @param data: numpy.array of data vectors (NxM, N = number of samples)
    returns Unitary arrays (NxN)
    '''
    #compute mean
    mean = np.mean(data, 0)
    new_data = data-mean

    U,S,V = np.linalg.svd(new_data.T)

    return U.T

def pca(data, k):
    '''
    TODO
    @param data: numpy.array of data vectors (NxM)
    @param k: number of eigenvectors to return

    returns (eigenvectors (NxM), eigenvalues (N))
    '''

    #compute mean
    #mean = np.mean(data, 0)
    mean = np.zeros(data.shape[1])[np.newaxis]
    for v in data:
        mean += v
    mean = mean / float(len(data))

    print(mean)
    
    new_data = data-mean

    #compute covariance matrix
    cov = np.dot(new_data.T, new_data) / new_data.shape[1]

    #compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(cov)
    
    #sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    return evecs.T[0:k], evals

def showVec(img, shape):
    '''
    GIVEN: 
    reshapes vector to given image size and plots it
    len(img) must be shape[0]*shape[1]
    @param img: given image as 1d vector 
    @param shape: shape of image
    '''
    img = np.reshape(img, shape)
    plt.imshow(img, cmap="gray")
    plt.show()

def normalized_linear_combination(vecs, weights):
    '''
    TODO
    Computes a linear combination of given vectors and weights.
    len(weights) must be <= vecs.shape[0]
    @param vecs: numpy.array of numpy.arrays to combine (NxM, N = number of basis vectors (unitary or eigenvectors))
    @param weights: list of weights (S) for the first S vectors
    returns numpy.array [M,1]
    '''
    vec = np.zeros(vecs.shape[1])[np.newaxis]
    for i in range(len(weights)):
        vec += weights[i] * vecs[i]
    vec = vec/float(len(weights))

    return vec

#load and scale down images
def load_dataset():
    '''
    Load data and transforms the values from [0,1] to [0,255]

    returns tuple of data (N,M) and shape of one img, N = number of samples, M = size of sample
    '''
    data = fetch_olivetti_faces().data

    data = np.round(data * 255)

    return data, (64, 64)

'''
TODO
1) Load dataset
2) compute principal components
3) compute linear combinations
4) display stuff
'''

#load dataset
data, shape = load_dataset()
print(data)

#do pca without svd
num_evecs = 500
evecs, evals = pca(data, num_evecs)

showVec(evecs[0], shape)
# needs integer division
showVec(evecs[evecs.shape[0]//2-1], shape)
showVec(evecs[evecs.shape[0]-1], shape)

w = [num_evecs-x for x in range(num_evecs)]

vec = normalized_linear_combination(evecs, w)
showVec(vec, shape)

w = evals[0:num_evecs]
vec = normalized_linear_combination(evecs, w)
showVec(vec, shape)

#do pca with svd
#U = pca_with_svd(data)
#showVec(U[0], shape)
#showVec(U[U.shape[0]/2-1], shape)

#vec = normalized_linear_combination(U, [1,1,1,1,1,1,1,1,1,1,1])
#showVec(vec, shape)

