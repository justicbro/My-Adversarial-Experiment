import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def nmf(V, rank, num_iterations=1000, tol=1e-4, device='cuda'):
    """
    Perform Non-negative Matrix Factorization (NMF) using alternating update method with PyTorch.

    Parameters:
    V (torch.Tensor): The non-negative input matrix to be factorized.
    rank (int): The rank of the factorization.
    num_iterations (int): The number of iterations to perform.
    tol (float): The tolerance for convergence.
    device (str): The device to run the calculations on ('cuda' or 'cpu').

    Returns:
    W (torch.Tensor): The non-negative matrix W.
    H (torch.Tensor): The non-negative matrix H.
    """
    # Initialize W and H with random non-negative values
    torch.manual_seed(0)
    W = torch.rand(V.shape[0], rank, device=device, dtype=torch.float32, requires_grad=True)
    H = torch.rand(rank, V.shape[1], device=device, dtype=torch.float32, requires_grad=True)

    for i in range(num_iterations):
        # Update H by fixing W
        H = H * (W.T @ V) / (W.T @ W @ H + 1e-10)
        H = torch.clamp(H, min=1e-10)  # Ensure non-negativity

        # Update W by fixing H
        W = W * (V @ H.T) / (W @ H @ H.T + 1e-10)
        W = torch.clamp(W, min=1e-10)  # Ensure non-negativity

        # Compute the approximation error
        error = torch.norm(V - W @ H, p='fro').item()
        if error < tol:
            break

        if i % 100 == 0:
            print(f"Iteration {i}, error: {error}")

    return W, H

def save_reconstructed_image(V, W, H, output_path):
    """
    Save the reconstructed image from the NMF result.

    Parameters:
    V (torch.Tensor): The original non-negative input matrix.
    W (torch.Tensor): The non-negative matrix W from NMF.
    H (torch.Tensor): The non-negative matrix H from NMF.
    output_path (str): The path to save the reconstructed image.
    """
    # Reconstruct the image
    V_reconstructed = torch.mm(W, H)

    # Clip values to be in the valid range [0, 255]
    V_reconstructed = torch.clamp(V_reconstructed, 0, 255)

    # Convert to uint8
    # 修改 save_reconstructed_image 函数中的 V_reconstructed 转换
    V_reconstructed = V_reconstructed.detach().cpu().numpy().astype(np.uint8)

    # Save the reconstructed image
    Image.fromarray(V_reconstructed).save(output_path)

def display_images(original, reconstructed, output_path):
    """
    Display the original and reconstructed images side by side and save the result.

    Parameters:
    original (torch.Tensor): The original image matrix.
    reconstructed (torch.Tensor): The reconstructed image matrix.
    output_path (str): The path to save the combined image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image
    axes[0].imshow(original.cpu().numpy(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display reconstructed image
    axes[1].imshow(reconstructed.detach().cpu().numpy(), cmap='gray')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    # Save the combined image
    plt.savefig(output_path)
    plt.show()

def main():
    # 读取 Lena 图片并转换为灰度图
    lena = Image.open('lena.png').convert('L')
    V = np.array(lena, dtype=np.float32)  # Ensure the dtype is float32

    # 将 V 转换为 PyTorch 张量并移动到 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    V = torch.tensor(V, device=device)

    # 指定分解的秩
    rank = 50

    # 执行 NMF
    W, H = nmf(V, rank, device=device)

    # 创建输出文件夹
    output_folder = 'result_NMF'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存重构后的图片
    output_path = os.path.join(output_folder, 'lena_reconstructed.png')
    save_reconstructed_image(V, W, H, output_path)

    # 显示并保存原图和重构后的图片
    combined_output_path = os.path.join(output_folder, 'lena_combined.png')
    V_reconstructed = torch.mm(W, H)
    # 修改 main 函数中的 V_reconstructed 转换
    V_reconstructed = torch.clamp(V_reconstructed, 0, 255).detach().cpu().numpy().astype(np.uint8)
    display_images(V, torch.tensor(V_reconstructed, device=device), combined_output_path)

    # 打印结果
    print("W:", W)
    print("H:", H)
    print("V ≈ WH:", W @ H)

if __name__ == "__main__":
    main()