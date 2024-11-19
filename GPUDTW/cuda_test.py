from __future__ import absolute_import
from __future__ import print_function

import numpy
import time


def k_means_cpu(S, n_clusters):
    k_means = KMeans(S, n_clusters=n_clusters).fit(S)
    pass

def compare_cpu_gpu(cpu, gpu):
    pass

if __name__ == '__main__':
    time_series_length = 1212
    n_series = 3
    n_clusters = 10
    S_nt = numpy.random.random ((n_series,n_clusters))
    S_nt = S_nt.astype(numpy.float32)

    cpu = k_means_cpu(S_nt, n_clusters)
    gpu = k_means_gpu(S_nt, n_clusters)
    compare_cpu_gpu(cpu, gpu)
