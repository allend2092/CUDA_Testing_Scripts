import numpy as np
import matplotlib.pyplot as plt
import time

# Pure Python implementation of the Mandelbrot set
def mandelbrot_python(xmin,xmax,ymin,ymax,width,height,max_iter):
    # Create linearly spaced values for x and y axes
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    # Initialize an empty array to store iteration counts for each pixel
    n3 = np.empty((width,height))
    # Iterate over each pixel in the image
    for i in range(width):
        for j in range(height):
            x = r1[i]
            y = r2[j]
            a, b = (0.0, 0.0)
            iter = 0
            # Calculate Mandelbrot iteration for the current pixel
            while (a*a + b*b <= 4.0 and iter < max_iter):
                a, b = a*a - b*b + x, 2.0*a*b + y
                iter += 1
            # Store the iteration count in the result array
            n3[i,j] = iter
    return (r1,r2,n3)

# Measure the execution time of the pure Python implementation
start_time = time.time()
d = mandelbrot_python(-2.0, 0.7, -1.35, 1.35, 1000, 1000, 256)
end_time = time.time()
print(f"Pure Python Time: {end_time - start_time} seconds")
# Display the generated Mandelbrot set
plt.imshow(d[2], extent=(d[0].min(), d[0].max(), d[1].min(), d[1].max()))
plt.show()

from numba import jit

# Numba JIT-compiled version of the Mandelbrot set (same as the pure Python version but faster due to JIT compilation)
@jit
def mandelbrot_numba(xmin,xmax,ymin,ymax,width,height,max_iter):
    # Same function body as mandelbrot_python
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            x = r1[i]
            y = r2[j]
            a, b = (0.0, 0.0)
            iter = 0
            while (a*a + b*b <= 4.0 and iter < max_iter):
                a, b = a*a - b*b + x, 2.0*a*b + y
                iter += 1
            n3[i,j] = iter
    return (r1,r2,n3)

# Measure the execution time of the Numba JIT-compiled implementation
start_time = time.time()
d = mandelbrot_numba(-2.0, 0.7, -1.35, 1.35, 1000, 1000, 256)
end_time = time.time()
print(f"Numba JIT Compilation Time: {end_time - start_time} seconds")
# Display the generated Mandelbrot set
plt.imshow(d[2], extent=(d[0].min(), d[0].max(), d[1].min(), d[1].max()))
plt.show()

from numba import cuda, float32

# GPU-accelerated version of the Mandelbrot set
@cuda.jit
def mandelbrot_gpu(min_x, max_x, min_y, max_y, image, iters):
    height, width = image.shape

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandelbrot(real, imag, iters)

# Device function to calculate Mandelbrot iteration for a single pixel (runs on the GPU)
@cuda.jit(device=True)
def mandelbrot(real, imag, max_iters):
    c = complex(real, imag)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return max_iters - 1  # Change this line

# Initialize an empty image array
gimage = np.zeros((1000, 1000), dtype=np.uint8)

# Define block and grid dimensions for GPU execution
blockdim = (32, 8)
griddim = (32, 16)

# Measure the execution time of the GPU-accelerated implementation
start_time_gpu = time.time()

# Transfer the image array to GPU memory
d_image = cuda.to_device(gimage)
# Execute the GPU function
mandelbrot_gpu[griddim, blockdim](-2.0, 0.7, -1.35, 1.35, d_image, 256)
# Transfer the result back to CPU memory
gimage = d_image.copy_to_host()

# Stop the timer
end_time_gpu = time.time()

# Display the GPU-generated Mandelbrot set
plt.imshow(gimage, extent=(-2.0, 0.7, -1.35, 1.35))
plt.show()

# Print the GPU execution time
print(f"GPU Execution Time: {end_time_gpu - start_time_gpu} seconds")
