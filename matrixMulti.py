

import numpy as np
matrix = [[2,0],[1,3]]  # 2,2
vector = [[4,5]] # 2,1

# [a1][a2] * [b1][b2] = [a1][b2]

# dot product
def dot(a,b):
    shape = np.shape(a)
    b = np.reshape(b,*shape)
    return np.dot(a,b)


# element-wise multiplication
def multi(a,b):
    if np.shape(a) == np.shape(b):
        return np.multiply(a,b)
    else:
        return "Input must be of the same shapes."


print(multi(vector,vector))
print(dot(matrix,vector))
