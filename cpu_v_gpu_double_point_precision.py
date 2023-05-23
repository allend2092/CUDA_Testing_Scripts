# My results when running this code:
# double-point precision

# CPU: AMD 7700x
# GPU: Nvidia 3080 10 GB

# 2.0.1+cu117
# Time taken to perform matrix multiplication on GPU: 36.05356550216675 seconds
# Time taken to perform matrix multiplication on CPU: 40.91802358627319 seconds
# Results are close:  True



import torch
import time

print(torch.__version__)

# Set the size of the square matrix
N = 20000

# Create the data arrays with double precision
A = torch.randn([N, N], device='cuda:0', dtype=torch.float64)
B = torch.randn([N, N], device='cuda:0', dtype=torch.float64)

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

# Check the results are close with a larger tolerance
print("Results are close: ", torch.allclose(C_cpu, C_gpu.to('cpu'), atol=1e-3))

