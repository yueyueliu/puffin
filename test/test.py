import torch.nn as nn
import torch

# 创建一个 Conv1d 层，用于声音特征提取
conv1d_layer = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
# 输入声音波形的维度为 (batch_size, num_channels, signal_length)
input_data = torch.randn(32, 1, 1000)  # 示例输入数据
# 通过 Conv1d 进行特征提取
output_features = conv1d_layer(input_data)

print(output_features.size())




# 创建一个 ConvTranspose1d 层，用于上采样
conv_transpose1d_layer = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=3)
# 输入低分辨率特征图的维度为 (batch_size, num_channels, feature_length)
input_features = torch.randn(32, 64, 100)  # 示例输入特征图
# 通过 ConvTranspose1d 进行上采样
upsampled_output = conv_transpose1d_layer(input_features)

print(upsampled_output.size())