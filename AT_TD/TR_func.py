import torch
import torch.autograd as autograd
import tensorly as tl
from tensorly.tenalg import mode_dot
from tensorly import unfold, fold
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def initialize_tr_cores(shape, ranks, device):
    cores = []
    for i in range(len(shape)):
        core_shape = (ranks[i], shape[i], ranks[i + 1])
        core = torch.randn(core_shape, device=device)
        cores.append(core)
    return cores

def show_images_pillow(image1, title1, image2, title2):
    # 转换为 PIL 格式
    img1 = Image.fromarray((image1 * 255).astype(np.uint8))  # 转换为 0-255 的整数值
    img2 = Image.fromarray((image2 * 255).astype(np.uint8))  # 转换为 0-255 的整数值

    # 显示图片
    img1.show(title=title1)
    img2.show(title=title2)


# 将 Tensorly backend 设置为 PyTorch
tl.set_backend('pytorch')

def generalized_k_contraction(X, Y, modes_X, modes_Y, k):
    """
    广义张量 k 收缩积的 GPU 加速实现

    参数:
        X (torch.Tensor): 输入张量 X
        Y (torch.Tensor): 输入张量 Y
        modes_X (list): X 中的模式索引，与 Y 的模式索引对齐
        modes_Y (list): Y 中的模式索引，与 X 的模式索引对齐

    返回:
        torch.Tensor: 收缩后的张量
    """

    # 将张量展开为矩阵
    X_unfolded_T = generalized_k_unfolding(X, modes_X, k).T  # 按第一个模式展开
    Y_unfolded = generalized_k_unfolding(Y, modes_Y, k)  # 按第一个模式展开

    # 对齐的模式进行矩阵乘积
    contracted = torch.matmul(X_unfolded_T, Y_unfolded)

    sorted_dims_X = [X.shape[i] for i in modes_X[k+1:]]
    sorted_dims_Y = [Y.shape[i] for i in modes_Y[k+1:]]
    # 将矩阵重新折叠回张量
    new_shape = sorted_dims_X + sorted_dims_Y
    # result = fold(contracted, mode=0, shape=new_shape)
    result = contracted.reshape(new_shape)

    return result





def generate_sequence(n, N):
    if n > N:
        raise ValueError("n must be less than or equal to N")
    return list(range(n+1, N+1)) + list(range(0, n))

def contract_except_n(cores, n):
    """
    计算除第 n 个核心张量之外的张量积
    """
    N = len(cores)
    # print("N", N)
    
    cont_seq = generate_sequence(n, N-1)
    # print("cont_seq", cont_seq)
  
    full_tensor = cores[cont_seq[0]]
    # print("full_tensor shape", full_tensor.shape)

    for i in range(1, N-1):
        dims = list(range(full_tensor.ndim))
        new_dims = [dims[-1]] + dims[:-1]
        core_i_dims = list(range(cores[cont_seq[i]].ndim))
        full_tensor = generalized_k_contraction(full_tensor, cores[cont_seq[i]], new_dims, core_i_dims, 0)
        # print("full_tensor shape", full_tensor.shape)

    # 将结果转换回 PyTorch 张量
    result = torch.tensor(full_tensor, requires_grad=True, dtype=torch.float32).to(cores[0].device)

    return result

def generalized_k_unfolding(X, modes, k):
    """
    广义张量 k 展开的 GPU 加速实现

    参数:
        X (torch.Tensor): 输入张量
        modes (list): 展开的模式顺序

    返回:
        torch.Tensor: 展开后的矩阵
    """
    # 验证输入参数
    if len(set(modes)) != len(modes):
        raise ValueError("模式索引不能重复！")
    
    if max(modes) >= X.ndim:
        raise ValueError("模式索引超出了张量的维度！")

    # 调整张量模式顺序
    permuted_X = X.permute(*modes)

    # 计算新的矩阵形状
    # print("prod X_k", torch.prod(torch.tensor([X.shape[i] for i in modes[:k]])).item())
    unfold_shape = (torch.prod(torch.tensor(permuted_X.shape[:k+1])).item(),
                    torch.prod(torch.tensor(permuted_X.shape[k+1:])).item())

    
    # 将调整后的张量 reshape 成矩阵
    unfolded_X = permuted_X.reshape(unfold_shape)

    return unfolded_X



def als_update_tr(X, cores, n, device):
    # 确保 X 是一个正确的张量
    X = tl.tensor(X, requires_grad=True, device=device)

    # 计算除 cores[n] 之外的张量积
    X_hat = contract_except_n(cores, n)

    # # 检查 contract_except_n 是否保持了梯度路径
    # print(f"X_hat requires_grad (n={n}):", X_hat.requires_grad)

    dims = list(range(X_hat.dim()))
    new_dims = [dims[-1]] + dims[:-1]
    X_hat_unfold = generalized_k_unfolding(X_hat, new_dims, 1)
    # X_hat_unfold_T = X_hat_unfold.T

    X_dims= list(range(X.dim()))
    new_X_dims = [X_dims[n]] + X_dims[n+1:] + X_dims[:n]
    X_unfold = generalized_k_unfolding(X, new_X_dims, 0)
    X_unfold = torch.tensor(X_unfold, requires_grad=True, dtype=torch.float32).to(device)
    # X_unfold_T = X_unfold.T


    new_core = X_unfold@X_hat_unfold.T@torch.linalg.pinv(X_hat_unfold@X_hat_unfold.T)
    # new_core = X_unfold @ X_hat_unfold.T @ torch.pinverse(X_hat_unfold @ X_hat_unfold.T)

    new_core = new_core.reshape(cores[n].shape[1], cores[n].shape[0], cores[n].shape[2])
    permuted_new_core = new_core.permute(1,0,2)
    return permuted_new_core

def tr_decompose(X, ranks, max_iter=500, tol=1e-10, device="cuda"):
    # 确保 X 是一个正确的张量
    X = tl.tensor(X, requires_grad=True, device=device)

    shape = list(X.shape)
    cores = initialize_tr_cores(shape, ranks, device)
    prev_loss = float('inf')
    
    for iteration in range(max_iter):
        for n in range(len(shape)):
            cores[n] = als_update_tr(X, cores, n, device)
        
        # 重建张量计算损失
        X_hat = tl.tr_to_tensor(cores)
        loss = torch.norm(X - X_hat) / torch.norm(X)
        
        if abs(prev_loss-loss) < tol:
            print(f"Converged at iteration {iteration + 1}")
            break

        if iteration % 10 == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss.item():.6f}")
        
        prev_loss = loss
    
    return cores

def tr_decompose2(X, cores, max_iter=500, tol=1e-10, dis_num = 10, device="cuda"):
    # 确保 X 是一个正确的张量
    X = tl.tensor(X, requires_grad=True, device=device)
    # print("X + E requires_grad:", X.requires_grad)

    shape = list(X.shape)
    prev_loss = float('inf')
    
    for iteration in range(max_iter):
        for n in range(len(shape)):
            cores[n] = als_update_tr(X, cores, n, device)
        
        # 重建张量计算损失
        X_hat = tl.tr_to_tensor(cores)
        loss = torch.norm(X - X_hat) / torch.norm(X)
        
        if abs(prev_loss-loss) < tol:
            print(f"Converged at iteration {iteration + 1}")
            break

        if iteration != 0 and iteration % dis_num == 0:
            print(f"Iteration {iteration + 1}, TRD_Loss: {loss.item():.6f}")
        
        prev_loss = loss
    
    return cores


def tensor_completion(X, mask, ranks, max_iter=500, tol=1e-10, device="cuda"):
    """
    Perform tensor completion using TR-ALS.
    
    Parameters:
    X: Incomplete tensor with missing values (PyTorch tensor)
    mask: Binary mask indicating observed entries (PyTorch tensor)
    ranks: List of ranks for the TR decomposition
    max_iter: Maximum number of ALS iterations
    tol: Convergence tolerance
    device: Device to perform computations on ("cuda" or "cpu")
    
    Returns:
    Completed tensor (PyTorch tensor)
    """
    # Initialize TR cores
    shape = list(X.shape)
    cores = initialize_tr_cores(shape, ranks, device)
    prev_loss = float('inf')
    
    for iteration in range(max_iter):
        for n in range(len(shape)):
            cores[n] = als_update_tr(X * mask, cores, n, device)
        
        # Reconstruct the tensor from TR cores
        X_hat = tl.tr_to_tensor(cores)
        
        # Compute the loss only on observed entries
        loss = torch.norm((X - X_hat) * mask) / torch.norm(X * mask)
        
        if abs(prev_loss - loss) < tol:
            print(f"Converged at iteration {iteration + 1}")
            break
        
        prev_loss = loss
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss.item()}")
    
    return X_hat

def reconstruct_tr(cores):
    """
    根据 TR 核心张量重构张量
    """
    N = len(cores)
    X_reconstructed = cores[0]
    for n in range(1, N):
        X_reconstructed = mode_dot(X_reconstructed, cores[n], mode=-1)
    return X_reconstructed