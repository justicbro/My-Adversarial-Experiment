import numpy as np
import tensorly as tl
from tensorly.decomposition import tensor_ring

# 设置张量的 backend 为 numpy
tl.set_backend('numpy')

# 生成一个随机张量
shape = (50, 60, 70)  # 张量维度
rank = [3, 4, 5, 3]  # TR 分解的秩，需要包含额外的末尾秩
tensor = np.random.random(shape)

# 进行 TR 分解
core_tensors = tensor_ring(tensor, rank=rank)

# 输出 TR 核心张量的信息
print("TR 分解的核心张量：")
for i, core in enumerate(core_tensors):
    print(f"G[{i}] shape: {core.shape}")

# 根据 TR 核心张量重构原始张量
reconstructed_tensor = tl.tr_to_tensor(core_tensors)

# 验证重构张量与原始张量的差异
error = np.linalg.norm(tensor - reconstructed_tensor) / np.linalg.norm(tensor)
print(f"重构误差: {error:.6f}")
