#from cupyx.scipy.spatial import distance_matrix
from numba import cuda
import csv
from timeit import default_timer as timer
import gen_data as gd
import numpy as np
import cupy as cp

@cuda.jit
def distance_matrix(d_arr, d_mat):
    i, j = cuda.grid(2)
    if i < d_arr.shape[0] and j < d_arr.shape[0] and i > j:
        d_mat[i, j] = ((d_arr[i, 0] - d_arr[j, 0]) ** 2 + (d_arr[i, 1] - d_arr[j, 1]) ** 2) ** 0.5

# Example usage
#gd.generate_random_points_csv(10000,'data.csv')
points_list = gd.read_csv_and_create_tuples('data.csv')
arr = np.array(points_list)
mat = np.empty((len(points_list),len(points_list)))
#print(arr)

# copy the array to gpu memory
d_arr = cuda.to_device(arr)
d_mat = cuda.to_device(mat)

#specify threads
threadsperblock = (16,16)
blockspergrid_x = (len(arr) + (threadsperblock[0] - 1)) // threadsperblock[0]
blockspergrid_y = (len(arr) + (threadsperblock[1] - 1)) // threadsperblock[1]
blockspergrid = (blockspergrid_x,blockspergrid_y)
#print(f'threadsperblock = {threadsperblock} and blockspergrid = {blockspergrid}')

s = timer()
mat = distance_matrix[blockspergrid, threadsperblock](d_arr, d_mat)
print(f'Total time: {timer()-s}')
#print(mat)

# create empty array
gpu_arr = np.empty(shape=d_mat.shape, dtype=d_mat.dtype)

# move data back to host memory
d_mat.copy_to_host(gpu_arr)

print("GPU Array: ", gpu_arr)

# Save the distance matrix to a CSV file
np.savetxt('distance_matrix.csv', gpu_arr, delimiter=',', fmt='%d')