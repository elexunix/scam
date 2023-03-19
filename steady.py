import torch

def SCAM_steady(objective_fn, init_params, num_steps=1000, learning_rate=0.01, gamma=0.5, sigma=0.5):
    """
    Implements the third-order Steady Curvature Aware Minimization algorithm to optimize a given objective function.

    Args:
    - objective_fn: a PyTorch function that takes in parameters and returns an objective value
    - init_params: a PyTorch tensor of initial parameters to optimize over
    - num_steps: an integer representing the number of optimization steps to take (default: 1000)
    - learning_rate: a float representing the learning rate of the optimizer (default: 0.01)
    - gamma: a float representing the curvature penalty parameter (default: 0.5)
    - sigma: a float representing the penalty parameter for deviation from the steady curvature path (default: 0.5)

    Returns:
    - params: a PyTorch tensor of optimized parameters
    - objective_val: the objective value at the optimized parameters
    """

    # Define optimization parameters
    p = init_params
    v = torch.zeros_like(p)
    g = torch.zeros_like(p)

    # Define functions for curvature and steady curvature
    def curvature(p):
        J = torch.autograd.functional.jacobian(objective_fn, p.unsqueeze(0))
        H = torch.autograd.functional.hessian(objective_fn, p.unsqueeze(0))
        K = torch.sum(torch.diagonal(torch.matmul(torch.matmul(J, torch.matmul(H.inverse(), torch.transpose(J, -1, -2))), torch.transpose(J, -1, -2)), dim1=-1, dim2=-2))
        return K

    def steady_curvature_path(p, v):
        J = torch.autograd.functional.jacobian(objective_fn, p.unsqueeze(0))
        Jv = torch.sum(torch.mul(J, v.unsqueeze(1)), dim=-1)
        v = v - gamma * torch.sum(Jv.unsqueeze(-1) * J, dim=1)
        v = v / torch.norm(v)
        return v

    # Initialize optimization loop
    for t in range(num_steps):
        # Compute objective value and gradients
        objective_val = objective_fn(p)
        objective_val.backward()
        g = p.grad

        # Compute curvature and steady curvature vector
        K = curvature(p)
        v = steady_curvature_path(p, v)

        # Compute acceleration term
        g_dot_v = torch.sum(torch.mul(g, v.unsqueeze(1)), dim=-1)
        a = -learning_rate * (g - ((1 - sigma) / K) * g_dot_v * v)

        # Update parameters
        p = p + a

        # Zero gradients
        p.grad.zero_()

    return p, objective_fn(p)
