import torch
import tensorly as tl
from PIL import Image
import numpy as np
from tensorly.decomposition import tensor_train
import TR_func as tr

tl.set_backend('pytorch')  # Set tensorly to use PyTorch backend

def objective_function(X, G_list, E):
    """
    Compute the objective function ||X - TN([G^(k)])||_F^2.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    """
    TN_G = tl.tr_to_tensor(G_list)  # Reconstruct tensor from tensor train
    return torch.norm(X + E - TN_G) ** 2

def optimize_perturbation(X, G_list, epsilon, lr=1e-3, num_steps=100):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    epsilon: Frobenius norm constraint for \mathcal{E}
    lr: Learning rate for gradient descent
    num_steps: Number of gradient descent steps
    """
    # Initialize perturbation tensor \mathcal{E}
    E = torch.zeros_like(X, requires_grad=True, device='cuda')

    optimizer = torch.optim.Adam([E], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Compute loss
        loss = objective_function(X, G_list, E)

        # Apply the constraint ||E||_F^2 <= epsilon
        if torch.norm(E) ** 2 > epsilon:
            E.data = epsilon * E / torch.norm(E)

        # Backpropagation
        loss.backward()

        # Update E
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    return E

# Example usage
if __name__ == "__main__":
    # Example tensor X (3D for simplicity)
    # 读取 Lena 图片
    lena = Image.open('lena.png')
    lena = np.array(lena)
    X = torch.tensor(lena/255.0, dtype=torch.float32)
    

    # Decompose X into tensor train format
    ranks = [10, 10, 10]  # Rank for each factor tensor
    ranks = ranks + [ranks[0]]  # 最后一个 rank 应该等于第一个 rank
    tr_cores = tr.tr_decompose(X, ranks, max_iter=100, tol=1e-10, device="cuda")

    # Set epsilon constraint
    epsilon = 1e-2

    # Optimize perturbation \mathcal{E}
    E_optimized = optimize_perturbation(X, tr_cores, epsilon)

    print("Optimized \mathcal{E}:", E_optimized)
