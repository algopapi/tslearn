import torch
from numba import cuda

a = torch.randn(10, device='cuda')
cuda_array = cuda.as_cuda_array(a)
print(cuda_array)