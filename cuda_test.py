import numpy
import pycuda
from tslearn.clustering import TimeSeriesKMeans

import pycuda
import pycuda.driver as cuda

def k_means_cpu(S, n_clusters):
    # Initialize CUDA context properly
    print(f"CUDA version {pycuda.VERSION}")
    print(f"Available GPU: {cuda.Device(0).name()}")
    k_means = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw").fit(S)
    return k_means

def k_means_gpu(S, n_clusters):
    cuda.init()
    print(f"Available GPU: {cuda.Device(0).name()}")
    k_means = TimeSeriesKMeans(S, n_clusters=n_clusters, metric="dtw_gpu").fit(S)
    return k_means

def compare_cpu_gpu(cpu, gpu):
    pass

if __name__ == '__main__':
    time_series_length = 1024
    n_series = 3
    n_clusters = 2
    S_nt = numpy.random.random ((n_series, time_series_length))
    S_nt = S_nt.astype(numpy.float32)

    cpu = k_means_cpu(S_nt, n_clusters)
    # gpu = k_means_gpu(S_nt, n_clusters)
    # compare_cpu_gpu(cpu, gpu)

