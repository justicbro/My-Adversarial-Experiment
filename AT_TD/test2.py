import torch
import tensorly as tl
from tensorly.decomposition import tensor_ring
from torch.optim import Adam

# 使用 PyTorch 作为后端
tl.set_backend('pytorch')

# 创建初始张量和扰动
torch.manual_seed(0)
original_tensor = torch.randn(70, 80, 90)  # 原始张量
rank = [2, 3, 2, 2]  # TR 分解的秩
perturbation = torch.zeros_like(original_tensor, requires_grad=True)  # 扰动项 \mathcal{E}

# 使用 tensorly 对张量进行 TR 分解
core_tensors = tensor_ring(original_tensor, rank=rank)  # 初始 TR 核心张量
core_tensors = [torch.tensor(core, requires_grad=True) for core in core_tensors]

# 优化器，更新扰动和核心张量
params = [perturbation] + core_tensors
optimizer = Adam(params, lr=1e-2)


# 损失函数
def loss_function(original, perturbation, cores):
    """
    计算重构误差的平方
    """
    reconstructed = tl.tr_to_tensor(cores)
    return torch.norm(original + perturbation - reconstructed) ** 2

# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # 计算损失
    loss = loss_function(original_tensor, perturbation, core_tensors)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    
    # 打印日志
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 最终结果
print("Final Loss:", loss_function(original_tensor, perturbation, core_tensors).item())
print("Learned perturbation:", perturbation)
for i, core in enumerate(core_tensors):
    print(f"Core[{i}] shape: {core.shape}")
