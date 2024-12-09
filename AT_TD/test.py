import numpy as np

import tensorly as tl
from tensorly.decomposition import tensor_ring

def initialize_tr(X, ranks):
    """
    初始化 TR 分解的核心张量
    Args:
        X: 输入张量，形状为 (I1, I2, ..., IN)
        ranks: TR 分解的秩，长度为 N+1，且 ranks[0] == ranks[-1] (闭环)

    Returns:
        G: 一个列表，包含 N 个核心张量，形状为 (R[n-1], I[n], R[n])
    """
    N = len(X.shape)
    G = [np.random.randn(ranks[n], X.shape[n], ranks[n+1]) for n in range(N)]
    return G

def reconstruct_tr(G):
    """
    根据 TR 核心张量重构张量
    """
    N = len(G)
    X_reconstructed = G[0]
    print("G[0] shape:", G[0].shape)
    for n in range(1, N):
        if n == N - 1:
            X_reconstructed = np.tensordot(X_reconstructed, G[n], axes=([-1, 1], [1, -1]))  # 环连接
            print("X_reconstructed shape after final ring connection:", X_reconstructed.shape)
            break

        print(f"G[{n}] shape:", G[n].shape)
        X_reconstructed = np.tensordot(X_reconstructed, G[n], axes=([-1], [0]))
        print("X_reconstructed shape:", X_reconstructed.shape)

    return X_reconstructed

def compute_gradient(X, E, G):
    """
    计算损失对 TR 核心张量和噪声张量的梯度
    """
    # 重构张量
    X_reconstructed = tl.tr_to_tensor(G)
    residual = (X + E) - X_reconstructed  # 计算残差

    # 对噪声张量的梯度
    dE = -2 * residual  # Frobenius 范数的导数

    # 对 TR 核心张量的梯度
    dG = []
    N = len(G)
    for n in range(N):
        # 构建左侧部分
        if n > 0:
            left_part = G[0]
            for m in range(1, n):
                left_part = np.tensordot(left_part, G[m], axes=([-1], [0]))
        else:
            left_part = None  # 如果没有左侧部分

        # 构建右侧部分
        if n < N - 1:
            right_part = G[n + 1]
            for m in range(n + 2, N):
                right_part = np.tensordot(right_part, G[m], axes=([-1], [0]))
        else:
            right_part = None  # 如果没有右侧部分

        # 计算当前核心张量的梯度
        if left_part is not None and right_part is not None:
            dG_n = np.tensordot(left_part, residual, axes=([-1], [0]))
            dG_n = np.tensordot(dG_n, right_part, axes=([-1], [0]))
        elif left_part is not None:
            dG_n = np.tensordot(left_part, residual, axes=([-1], [0]))
        elif right_part is not None:
            print("n:", n)
            print("residual shape:", residual.shape)
            print("right_part shape:", right_part.shape)

            dG_n = np.tensordot(residual, right_part, axes=([0], [0]))
        else:
            dG_n = residual

        dG.append(-2 * dG_n)  # Frobenius 范数的导数

    return dE, dG



def update_parameters(E, G, dE, dG, lr):
    """
    更新噪声张量和核心张量
    Args:
        E: 噪声张量
        G: TR 核心张量列表
        dE: 噪声张量的梯度
        dG: 核心张量的梯度列表
        lr: 学习率

    Returns:
        更新后的 E 和 G
    """
    E -= lr * dE
    G = [G[n] - lr * dG[n] for n in range(len(G))]
    return E, G

# 示例流程
np.random.seed(42)
I = [5, 6, 7]  # 张量的形状
ranks = [2, 3, 4, 2]  # TR 秩 (闭环)
X = np.random.randn(*I)  # 输入张量
E = np.random.randn(*I) * 0.01  # 初始化噪声张量
G = initialize_tr(X, ranks)  # 初始化核心张量

# 训练参数
epochs = 100
lr = 0.01

for epoch in range(epochs):
    # 计算梯度
    dE, dG = compute_gradient(X, E, G)
    print("Gradient computation successful!")

    # 更新参数
    E, G = update_parameters(E, G, dE, dG, lr)
    
    # 打印损失
    reconstructed_X = tl.tr_to_tensor(G)
    loss = np.linalg.norm((X + E) - reconstructed_X)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

print("训练完成！")
