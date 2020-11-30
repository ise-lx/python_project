import numpy as np

x = [[1,2],[3,5]]
x = np.asarray(x)
y = [[0,5],[7,7]]
y = np.asarray(y)

num = x.shape[0]
dists = np.zeros((num, num))




# dists = compute_distances_no_loops(x,y)
print(dists)



