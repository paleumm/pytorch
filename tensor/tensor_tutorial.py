import torch
import numpy as np

data = [[1, 2], [3, 4]]
# init from data
x_data = torch.tensor(data)
# print(x_data)

# from numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# print(x_np)

# from another tensor
x_ones = torch.ones_like(
    x_data
)  # same properties with x_data but every elements is '1'
# print(f"Ones Tensor: \n {x_ones}")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatypes
# print(f"Random Tensor: \n {x_rand}")


shape = (2, 3)
rand_tensor = torch.rand(shape)  # == rand(2,3)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(rand_tensor)
# print(ones_tensor)
# print(zeros_tensor)

# Attributes
tensor = torch.rand(3, 4)

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(
#    f"Device tensor is stored on: {tensor.device}"
# )  # which device tensor will be allocate

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
# print(f"Device tensor is stored on: {tensor.device}")


# indexing and slicing
tensor = torch.rand(4, 4)
# print(f"First row: {tensor[0]}")
# print(f"First column: {tensor[:, 0]}")
# print(f"Last column: {tensor[..., 1]}")

# set all elements in column 1 to 0
tensor[:, 1] = 0
# print(tensor)


# joining the tensor
t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)


# Arithmetic operations

# matrix multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# print(y1)
# print(y2)
# print(y3)


# element-wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


# print(z1)
# print(z2)
# print(z3)

# single-element tensor
agg = tensor.sum()
agg_item = agg.item()
# print(agg_item, type(agg_item))
# print(agg)

# In-place operations
# print(tensor)
tensor.add_(5)  # add 5 to each element
# print(tensor)


# Bridge with Numpy
t = torch.ones(5)
# t: tensor([1., 1., 1., 1., 1.])
n = t.numpy()
# n: [1. 1. 1. 1. 1.]

# A change in the tensor reflects in the NumPy array.
t.add_(1)
# t: tensor([2., 2., 2., 2., 2.])
# n: [2. 2. 2. 2. 2.]


# Numpy array to tensor

n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor.

np.add(n, 1, out=n)
# t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
# n: [2. 2. 2. 2. 2.]
