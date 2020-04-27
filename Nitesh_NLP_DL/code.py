# --------------
# Code starts here

import numpy as np

# Code starts here

# Adjacency matrix
adj_mat = np.array([[0,0,0,0,0,0,1/3,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                   [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                  [0,0,1/2,1/3,0,0,1/3,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/3,0]])

# Compute eigenvalues and eigencevectrs

eigenvalues , eigencevectors = np.linalg.eig(adj_mat)

print(eigenvalues)

print(eigencevectors)




# Eigen vector corresponding to 1

eigen_1 = abs(eigencevectors[:,0])

print(eigen_1)


eigen_1 = abs(eigencevectors[:,0])/np.linalg.norm(eigencevectors[:,0],1)

print(eigen_1)

# most important page

page = np.where(np.max(eigen_1)== eigen_1)[0][0] + 1
print(page)




# Code ends here


# --------------
# Code starts here

adj_mat
print(adj_mat)



# Initialize stationary vector I

init_I = np.array([1,0,0,0,0,0,0,0])
print(init_I)

a= np.dot(adj_mat , init_I)
print(a)

b = np.linalg.norm(init_I , 1)
print(b)


for i in range(10):
        init_I = np.dot(adj_mat , init_I)
        init_I /= np.linalg.norm(init_I , 1)

print(init_I)



# Perform iterations for power method

power_page = np.where(np.max(init_I)==init_I)[0][0] +1
print(power_page)


# Code ends here


# --------------
# Code starts here

# New Adjancency matrix
# New Adjancency matrix
new_adj_mat = np.array([[0,0,0,0,0,0,0,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                  [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                   [0,0,1/2,1/3,0,0,1/2,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/2,0]])

# Initialize stationary vector I

new_init_I = np.array([1,0,0,0,0,0,0,0])

print(new_init_I)


# Perform iterations for power method
for _ in range(10):
    new_init_I = np.dot(new_adj_mat, new_init_I)
    new_init_I /= np.linalg.norm(new_init_I, 1)


print(new_init_I)


power_page = np.where(np.max(new_init_I) == new_init_I)[0][0] + 1
print(power_page)

print(new_init_I)

    




# Perform iterations for power method





# Code ends here


# --------------
# Alpha value
alpha = 0.85

new_adj_mat
print(new_adj_mat)


len(new_adj_mat)

np.ones(new_adj_mat.shape)

G = (alpha*new_adj_mat) + (1-alpha)*(1 / len(new_adj_mat))*np.ones(new_adj_mat.shape)



print(G)

final_init_I = np.array([1 , 0 , 0,0,0,0,0,0])
print(final_init_I)



for _ in range(1000):
    final_init_I = np.dot(G, final_init_I)
    final_init_I /= np.linalg.norm(final_init_I, 1)


print(final_init_I)


power_page = np.where(np.max(final_init_I) == final_init_I)[0][0] + 1
print(power_page)

print(final_init_I)





# Code starts here

# Modified adjancency matrix


# Initialize stationary vector I


# Perform iterations for power method


# Code ends here


