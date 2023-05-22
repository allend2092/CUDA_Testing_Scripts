import torch
import time


print(torch.__version__)


# Set the size of the square matrix
N = 20000

# Create the data arrays
A = torch.randn([N, N], device='cuda:0')
B = torch.randn([N, N], device='cuda:0')

# Perform the matrix multiplication on the GPU and measure the time taken
start = time.time()
C_gpu = torch.matmul(A, B)
torch.cuda.synchronize() # Ensure the GPU operations are finished
end = time.time()
gpu_time = end - start
print(f"Time taken to perform matrix multiplication on GPU: {gpu_time} seconds")

# Move the data to the CPU
A = A.to('cpu')
B = B.to('cpu')

# Perform the matrix multiplication on the CPU and measure the time taken
start = time.time()
C_cpu = torch.matmul(A, B)
end = time.time()
cpu_time = end - start
print(f"Time taken to perform matrix multiplication on CPU: {cpu_time} seconds")

# Check the results are close
print("Results are close: ", torch.allclose(C_cpu, C_gpu.to('cpu'), atol=1e-3))











