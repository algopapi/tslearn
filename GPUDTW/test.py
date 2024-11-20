from __future__ import absolute_import
from __future__ import print_function

import numpy
import time

import pycuda.driver as cuda

from GPUDTW import cuda_dtw
from GPUDTW import cpu_dtw, dtw_1D_jit2

if __name__ == '__main__':
    n_series = 1
    series_length = 1022
    n_clusters = 4

    S = numpy.random.random ((n_series,series_length))
    S = S.astype(numpy.float32)

    T = numpy.random.random ((n_clusters,series_length))
    T = T.astype(numpy.float32)

    t0 = time.time()
    ret_cuda = cuda_dtw (S, T)
    print(ret_cuda)
    print(ret_cuda.shape) # 3, 1311 , i.e., n=3 series, c=1312 distances 
    assert ret_cuda.shape == (1, 4), "not what we expected"
    print ("cuda time:",time.time()-t0)
