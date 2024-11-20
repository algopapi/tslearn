import time
import numpy
import pycuda
from tslearn.clustering import TimeSeriesKMeans

import pycuda
import pycuda.driver as cuda

def k_means_cpu(S, n_clusters):
    # Initialize CUDA context properly
    cuda.init()
    cpu_time_start = time.time()
    k_means = TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw").fit(S)
    print(f"CPU time: {time.time()- cpu_time_start}")
    return k_means

def k_means_gpu(S, n_clusters):
    gpu_time_start = time.time()
    
    print(f"Available GPU: {cuda.Device(0).name()}")
    k_means = TimeSeriesKMeans(n_clusters=n_clusters, metric="gpudtw").fit(S)
    print(f"GPU time: {time.time() - gpu_time_start}")
    return k_means

def compare_cpu_gpu(cpu, gpu):
    pass

if __name__ == '__main__':
    time_series_length = 1024
    n_series = 1000 
    n_clusters = 10
    S_nt = numpy.random.random ((n_series, time_series_length))
    S_nt = S_nt.astype(numpy.float32)

    cpu = k_means_cpu(S_nt, n_clusters)
    # cpu time = 525

    # gpu = k_means_gpu(S_nt, n_clusters)
    # compare_cpu_gpu(cpu, gpu)