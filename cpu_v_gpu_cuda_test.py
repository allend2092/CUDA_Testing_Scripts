import numpy as np
import time
from numba import cuda, float32

# Set the size of the square matrix
N = 23000

# Define a CUDA kernel that performs matrix multiplication
@cuda.jit
def matmul_gpu(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

# Create the data arrays
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# Allocate memory on the GPU and copy the data over
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)
C_global_mem = cuda.device_array((N, N))

# Set up the grid and blocks
threadsperblock = (32, 32)
blockspergrid_x = int(np.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Perform the matrix multiplication on the GPU and measure the time taken
start = time.time()
matmul_gpu[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
cuda.synchronize() # Ensure the GPU operations are finished
end = time.time()
gpu_time = end - start
print(f"Time taken to perform matrix multiplication on GPU: {gpu_time} seconds")

# Perform the matrix multiplication on the CPU and measure the time taken
start = time.time()
C_cpu = np.matmul(A, B)
end = time.time()
cpu_time = end - start
print(f"Time taken to perform matrix multiplication on CPU: {cpu_time} seconds")

# Copy the result from the GPU back to the CPU
C_gpu = C_global_mem.copy_to_host()

# Check the results are close
print("Results are close: ", np.allclose(C_cpu, C_gpu, atol=1e-5))
