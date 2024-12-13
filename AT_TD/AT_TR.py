import torch
import tensorly as tl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorly.decomposition import tensor_train
import TR_func as tr
import logging
logging.basicConfig(filename='AT_TR_compare.log', level=logging.INFO, format='%(asctime)s - %(message)s')

tl.set_backend('pytorch')  # Set tensorly to use PyTorch backend

class OptimizationConfig:
    def __init__(self, epsilon, lr, outer_num_steps, inner_num_steps, outer_dis_num, inner_dis_num, inner_tol):
        self.epsilon = epsilon
        self.lr = lr
        self.outer_num_steps = outer_num_steps
        self.inner_num_steps = inner_num_steps
        self.outer_dis_num = outer_dis_num
        self.inner_dis_num = inner_dis_num
        self.inner_tol = inner_tol

def objective_function_ATTR(X, G_list, E, iner_num_steps = 100, inner_tol = 1e-10):
    """
    Compute the objective function ||X - TN([G^(k)])||_F^2.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    """
    tr_cores = tr.tr_decompose2(X+E, G_list, max_iter=iner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")

    # 确保 tr_cores 启用了梯度追踪
    for core in tr_cores:
        core.requires_grad_(True)

    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train
    return -torch.norm(X - TN_G) ** 2, TN_G

def objective_function_ATR(X, G_list, E, inner_num_steps, inner_tol):
    """
    Compute the comparison objective function.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    inner_num_steps: Number of inner optimization steps
    inner_tol: Tolerance for inner optimization
    """
    # Example: Use a different decomposition method or a different loss function
    tr_cores = tr.tr_decompose2(X, G_list, max_iter=inner_num_steps, tol=inner_tol, device="cuda")
    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train
    return -torch.norm(X+E - TN_G) ** 2, TN_G  # Example: Minimize the norm instead of maximizing


def optimize_perturbation(X, G_list, config: OptimizationConfig):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    epsilon: Frobenius norm constraint for \mathcal{E}
    lr: Learning rate for gradient descent
    num_steps: Number of gradient descent steps
    """
    # Initialize perturbation tensor \mathcal{E}
    E = torch.randn_like(X, requires_grad=True, device='cuda')

    optimizer = torch.optim.Adam([E], lr=config.lr)

    for step in range(config.outer_num_steps):
        optimizer.zero_grad()

        # Compute loss
        loss, TN_G = objective_function_ATTR(X, G_list, E, config.inner_num_steps, config.inner_tol)

        # Apply the constraint ||E||_F^2 <= epsilon
        if torch.norm(E) ** 2 > config.epsilon:
            E.data = config.epsilon * E / torch.norm(E)

        # Backpropagation
        loss.backward()

        # Update E
        optimizer.step()

        if step % config.outer_dis_num == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    return E, TN_G

def optimize_perturbation_comparison(X, G_list, config: OptimizationConfig):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent for comparison.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    config: Optimization configuration
    """
    # Initialize perturbation tensor \mathcal{E}
    E = torch.randn_like(X, requires_grad=True, device='cuda')

    optimizer = torch.optim.Adam([E], lr=config.lr)

    for step in range(config.outer_num_steps):
        optimizer.zero_grad()

        # Compute loss
        loss, TN_G = objective_function_ATR(X, G_list, E, config.inner_num_steps, config.inner_tol)

        # Apply the constraint ||E||_F^2 <= config.epsilon
        if torch.norm(E) ** 2 > config.epsilon:
            E.data = config.epsilon * E / torch.norm(E)

        # Backpropagation
        loss.backward()

        # Update E
        optimizer.step()

        if step != 0 and step % config.outer_dis_num == 0:
            print(f"Comparison Step {step}, Loss: {loss.item()}")

    return E, TN_G

# Example usage
if __name__ == "__main__":
    # Example tensor X (3D for simplicity)
    # 读取 Lena 图片
    lena = Image.open('lena.png')
    lena = np.array(lena)
    X = torch.tensor(lena/255.0, dtype=torch.float32).to("cuda")
    

    # Decompose X into tensor train format
    ranks = [10, 10, 10]  # Rank for each factor tensor
    ranks = ranks + [ranks[0]]  # 最后一个 rank 应该等于第一个 rank
    tr_cores = tr.tr_decompose(X, ranks, max_iter=1, tol=1e-10, device="cuda")

    # Set input parameter 
    epsilon = 1e2 # Frobenius norm constraint for perturbation tensor
    inner_tol = 1e-10 # Tolerance for inner optimization
    lr = 0.01  # Learning rate for gradient descent
    outer_num_steps = 300 # Number of outer optimization steps
    inner_num_steps = 1000 # Number of inner optimization steps
    outer_dis_num = 10 # Display loss every outer_dis_num steps
    inner_dis_num = 2000 # Display loss every inner_dis_num steps

    config = OptimizationConfig(epsilon, lr, outer_num_steps, inner_num_steps, outer_dis_num, inner_dis_num, inner_tol)

    # Optimize perturbation \mathcal{E}
    E_optimized, TN_G = optimize_perturbation(X, tr_cores, config)

    # Optimize perturbation \mathcal{E} using comparison algorithm
    E_comparison, TN_G_comparison = optimize_perturbation_comparison(X, tr_cores, config)

    # 计算 norm2
    norm2_our_algorithm = torch.norm(X - TN_G).item()
    norm2_comparison_algorithm = torch.norm(X - TN_G_comparison).item()

    # 打印 norm2 结果
    print(f"Norm2 (Our Algorithm): {norm2_our_algorithm}")
    print(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm}")

    # 记录 norm2 结果到日志
    logging.info(f"Norm2 (Our Algorithm): {norm2_our_algorithm}")
    logging.info(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm}")

    # 将张量从 CUDA 移动到 CPU 并转换为 NumPy 数组
    X_cpu = X.cpu().detach().numpy()
    E_cpu = E_optimized.cpu().detach().numpy()
    X_E_cpu = (X + E_optimized).cpu().detach().numpy()
    TN_G = (TN_G).cpu().detach().numpy()
    E_comparison_cpu = E_comparison.cpu().detach().numpy()
    X_E_comparison_cpu = (X + E_comparison).cpu().detach().numpy()
    TN_G_comparison = (TN_G_comparison).cpu().detach().numpy()


    # 创建一个全白的图像
    white_image = np.ones_like(X_cpu)

    # 画图
    fig, axes = plt.subplots(2, 4, figsize=(15, 5))

    # 原图
    axes[0, 0].imshow(X_cpu)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 噪声图
    axes[0, 1].imshow(E_cpu)
    axes[0, 1].set_title('Adversarial Noise')
    axes[0, 1].axis('off')

    # 加噪后的图
    axes[0, 2].imshow(X_E_cpu)
    axes[0, 2].set_title('Adversarial Noisy Image')
    axes[0, 2].axis('off')

    # 加噪后的图
    axes[0, 3].imshow(TN_G)
    axes[0, 3].set_title('Adversarial Noisy Image')
    axes[0, 3].axis('off')

    # 白色图像
    axes[1, 0].imshow(white_image)
    axes[1, 0].set_title('White Image')
    axes[1, 0].axis('off')

    # 对比算法的噪声图
    axes[1, 1].imshow(E_comparison_cpu)
    axes[1, 1].set_title('Comparison Noise')
    axes[1, 1].axis('off')

    # 对比算法的加噪后的图
    axes[1, 2].imshow(X_E_comparison_cpu)
    axes[1, 2].set_title('Comparison Noisy Image')
    axes[1, 2].axis('off')

    # 加噪后的图
    axes[1, 3].imshow(TN_G_comparison)
    axes[1, 3].set_title('Adversarial Noisy Image')
    axes[1, 3].axis('off')

    # 保存图片到文件
    plt.savefig('result/lena_comparison2.png')
    plt.show()
    