import torch
import tensorly as tl
from tensorly.tenalg import mode_dot
from tensorly import unfold, fold
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def np_to_tensor(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)

def np_to_var(img_np, dtype = torch.cuda.FloatTensor):
    '''Converts image in numpy.array to torch.Variable.
    
    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(np_to_tensor(img_np)[None, :])

def var_to_np(img_var):
    '''Converts an image in torch.Variable format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.data.cpu().numpy()[0]

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

import torch
import tensorly as tl
from tensorly.base import unfold, fold

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
    # 检查张量维度匹配
    if len(modes_X) != len(modes_Y):
        raise ValueError("模式数目必须匹配！")

    dims = list(range(X.dim()))
    X_dims = dims
    Y_dims = [dims[-1]] + dims[:-1]
    # 将张量展开为矩阵
    X_unfolded_T = generalized_k_unfolding(X, modes_X, k).T  # 按第一个模式展开
    Y_unfolded = generalized_k_unfolding(Y, modes_Y, k)  # 按第一个模式展开

    # 对齐的模式进行矩阵乘积
    contracted = torch.matmul(X_unfolded_T, Y_unfolded)

    # 将矩阵重新折叠回张量
    new_shape = list(X.shape[:modes_X[0]]) + list(Y.shape[modes_Y[-1]+1:])
    result = fold(contracted, mode=0, shape=new_shape)

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
        full_tensor = generalized_k_contraction(full_tensor, cores[cont_seq[i]], [-1], [0])
        # print("full_tensor shape", full_tensor.shape)

    # 将结果转换回 PyTorch 张量
    result = torch.tensor(full_tensor, dtype=torch.float32).to(cores[0].device)

    return result


import torch
import tensorly as tl

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
    unfold_shape = (torch.prod(torch.tensor([X.shape[i] for i in modes[:k]])).item(),
                    -1)
    
    # 将调整后的张量 reshape 成矩阵
    unfolded_X = permuted_X.reshape(unfold_shape)

    return unfolded_X



def als_update_tr(X, cores, n, device):
    # 确保 X 是一个正确的张量
    X = tl.tensor(X, device=device)
    # 计算除 cores[n] 之外的张量积
    X_hat = contract_except_n(cores, n)
    dims = list(range(X_hat.dim()))
    new_dims = [dims[-1]] + dims[:-1]
    X_hat_unfold = generalized_k_unfolding(X_hat, new_dims, 2)
    X_hat_unfold_T = X_hat_unfold.T

    # 计算新的 cores[n]
    X_unfold = unfold(X, n)
    X_unfold = torch.tensor(X_unfold, dtype=torch.float32).to(device)
    X_unfold_T = X_unfold.T

    new_core = torch.linalg.lstsq(X_hat_unfold_T, X_unfold_T).solution 
    new_core = new_core.T

    new_core = new_core.reshape(cores[n].shape[1], cores[n].shape[0], cores[n].shape[2])
    permuted_new_core = new_core.permute(*[1,0,2])
    return permuted_new_core

def tr_decompose(X, ranks, max_iter=500, tol=1e-10, device="cuda"):
    # 确保 X 是一个正确的张量
    X = tl.tensor(X, device=device)

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

def reconstruct_tr(cores):
    """
    根据 TR 核心张量重构张量
    """
    N = len(cores)
    X_reconstructed = cores[0]
    for n in range(1, N):
        X_reconstructed = mode_dot(X_reconstructed, cores[n], mode=-1)
    return X_reconstructed

if __name__ == "__main__":

    # 读取 Lena 图片
    lena = Image.open('lena.png')
    lena = np.array(lena)
    lena = torch.tensor(lena, dtype=torch.float32)

    # 输入张量和秩
    shape = lena.shape
    print("lena_shape", shape)
    ranks = [2, 5, 6, 2]  # 注意，ranks 的长度应该是 len(shape) + 1

    # TR 分解
    tr_cores = tr_decompose(lena, ranks, max_iter=500, tol=1e-10, device="cuda")
    print("TR decomposition completed!")
    for i, core in enumerate(tr_cores):
        print(f"Core {i}: Shape {core.shape}")

    # 重构张量
    lena_reconstructed = tl.tr_to_tensor(tr_cores).cpu().numpy()


    # 显示原始和重构的图片
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Lena')
    plt.imshow(lena.cpu().numpy()/255, cmap='gray')
    plt.show()
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Lena')
    plt.imshow(lena_reconstructed/255, cmap='gray')

    # 保存图片到文件
    plt.savefig('lena_comparison.png')

    plt.show()

    