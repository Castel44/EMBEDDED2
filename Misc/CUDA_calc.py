from hpelm.nnets.slfn import SLFN
import numpy as np
from scipy.linalg import lapack
from scipy.spatial.distance import cdist
from pycuda import gpuarray, cumath, autoinit
from pycuda.compiler import SourceModule
from skcuda import linalg, misc, cublas
import pycuda.autoinit


def _dev_lin(X, B):
    """Linear function on GPU.

    Returns:
        devH (np.array): GPU matrix with the result.
    """

    linalg.init()
    devX = gpuarray.to_gpu(np.asarray(X, dtype=np.float32, order='C'))
    devB = gpuarray.to_gpu(np.asarray(B, dtype=np.float32, order='C'))
    devH = linalg.dot(devX, devB)
    return devH.get()


def _dev_sigm(X, B):
    """Compute Sigmoid on GPU for a given array and return array.
    Returns:
        devH (np.array): Numpy matrix with the result."""

    linalg.init()
    kernel = """
        __global__ void dev_sigm(%s *a) {
            unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
            a[idx] = 1.0 / ( exp(a[idx]) + 1 );
        }
        """
    kernel = kernel % "float"
    dev_sigm = SourceModule(kernel).get_function("dev_sigm")
    dev_sigm.prepare("P")

    devX = gpuarray.to_gpu(np.asarray(X, dtype=np.float32, order='C'))
    devB = gpuarray.to_gpu(np.asarray(B, dtype=np.float32, order='C'))
    devH = linalg.dot(devX, devB)
    block = devH._block
    grid = (int(np.ceil(1.0 * np.prod(devH.shape) / block[0])), 1)
    dev_sigm.prepared_call(grid, block, devH.gpudata)
    return devH.get()


def _dev_sigm_stable(X, B):
    """Compute Sigmoid on GPU for a given array and return array.
    Uses the exp trick, should be more numerically stable. [x = exp(log(x))]
    Returns:
        devH (np.array): Numpy matrix with the result."""

    linalg.init()
    kernel = """
        __global__ void dev_sigm(%s *a) {
            unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;            
            a[idx] = exp(-log(1.0 + exp(a[idx])));
        }
        """
    kernel = kernel % "float"
    dev_sigm = SourceModule(kernel).get_function("dev_sigm")
    dev_sigm.prepare("P")

    devX = gpuarray.to_gpu(np.asarray(X, dtype=np.float32, order='C'))
    devB = gpuarray.to_gpu(np.asarray(B, dtype=np.float32, order='C'))
    devH = linalg.dot(devX, devB)
    block = devH._block
    grid = (int(np.ceil(1.0 * np.prod(devH.shape) / block[0])), 1)
    dev_sigm.prepared_call(grid, block, devH.gpudata)
    return devH.get()


def _dev_tanh(X, B):
    """Hyperbolic tangent function on GPU.

    Returns:
        devH (np.array): Numpy matrix with the result.
    """

    linalg.init()
    devX = gpuarray.to_gpu(np.asarray(X, dtype=np.float32, order='C'))
    devB = gpuarray.to_gpu(np.asarray(B, dtype=np.float32, order='C'))
    # devH = gpuarray.empty((X.shape[0], B.shape[1]), dtype=np.float32)
    devH = linalg.dot(devX, devB)
    cumath.tanh(devH, out=devH)
    return devH.get()
