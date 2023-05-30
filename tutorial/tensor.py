import torch
import numpy as np

# output
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(f"x_data: {x_data}")

# ランダム値や定数のテンソルの作成
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# tensorの操作
tensor = torch.ones(4, 4)
print("First row: ", tensor[0])
print("First column: ", tensor[:, 0])
print("Last column:", tensor[..., -1])
tensor[:, 1] = 0
print(tensor)
