from __future__ import absolute_import
from __future__ import print_function

import numpy
import time

import pycuda.driver as cuda

from GPUDTW import cuda_dtw
from GPUDTW import cpu_dtw, dtw_1D_jit2

if __name__ == '__main__':
    cuda.init()
    S = numpy.random.random ((3,1024))
    # k_means = TimeSeriesKMeans(S, n_clusters=n_clusters, metric="dtw").f)it(S)
    S = S.astype(numpy.float32)
    T = numpy.random.random ((1312,1024))
    T = T.astype(numpy.float32)

    t0 = time.time()
    ret_cuda = cuda_dtw (S, T)
    print(ret_cuda)
    print(ret_cuda.shape)
    print ("cuda time:",time.time()-t0)
    # cuda_verify = numpy.sqrt((ret_cuda - ret_cpu)**2)
    # print ("Maximum Deviation in cuda with CPU ", cuda_verify.max())