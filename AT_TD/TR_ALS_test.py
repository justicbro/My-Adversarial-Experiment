import torch
import tensorly as tl
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import TR_func as tr


if __name__ == "__main__":

    # 读取 Lena 图片
    lena = Image.open('lena.png')
    lena = np.array(lena)
    lena = torch.tensor(lena/255.0, dtype=torch.float32)
    # lena = torch.randn(30,40,50,60, device="cuda")

    # 输入张量和秩
    shape = lena.shape
    print("lena_shape", shape)
    ranks = [10, 10, 10]  # 注意，ranks 的长度应该是 len(shape) + 1
    ranks = ranks + [ranks[0]]  # 最后一个 rank 应该等于第一个 rank

    # TR 分解
    tr_cores = tr.tr_decompose(lena, ranks, max_iter=1000, tol=1e-10, device="cuda")
    print("TR decomposition completed!")
    for i, core in enumerate(tr_cores):
        print(f"Core {i}: Shape {core.shape}")

    # 重构张量
    lena_reconstructed = tl.tr_to_tensor(tr_cores).cpu().numpy()


    # 显示原始和重构的图片
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Lena')
    plt.imshow(lena.cpu().numpy(), cmap='gray')
    plt.show()
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Lena')
    plt.imshow(lena_reconstructed, cmap='gray')

    # 保存图片到文件
    plt.savefig('lena_comparison.png')

    plt.show()

    