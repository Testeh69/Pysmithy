import numpy as np
import time
import smithy as sm


a = np.random.rand(1000, 1000)  # Random 1000x1000 matrix
b = np.random.rand(1000, 1000)  # Random 1000x1000 matrix

# Start timing the operation
start_time = time.time()
# Perform matrix multiplication
result = np.dot(a, b)
# Stop timing
end_time = time.time()

print("Matrix multiplication with numpy took", end_time - start_time, "seconds.")

start_time = time.time()
# Perform matrix multiplication
result = sm.dot_matrice(a, b)
# Stop timing
end_time = time.time()


print("Matrix multiplication with smithy took", end_time - start_time, "seconds.")
