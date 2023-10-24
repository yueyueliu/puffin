import torch

# 创建一个简单的示例输入张量 x
batch_size = 2
channels = 4
sequence_length = 10
x = torch.randn(batch_size, channels, sequence_length)

# 首先执行 x.flip([1, 2])，翻转第 1 维和第 2 维
x_flipped_once = x.flip([1, 2])

# 接着再执行 x.flip([2])，翻转第 2 维
x_flipped_twice = x_flipped_once.flip([2])

print("Original x:")
print(x)
print("x flipped once:")
print(x_flipped_once)
print("x flipped twice:")
print(x_flipped_twice)
