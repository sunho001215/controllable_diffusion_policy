import torch

def compute_PMP(x_t, timestep, model, noise_scheduler):
    device = x_t.get_device()

    # Get alpha_t
    alpha_t = noise_scheduler.alphas_cumprod[timestep]
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

    # Predict noise
    epsilon_theta = model(x_t, torch.tensor([timestep], dtype=torch.long).to(device))

    # Calcaulte PMP
    PMP = (x_t - sqrt_one_minus_alpha_t * epsilon_theta) / sqrt_alpha_t

    return PMP

def compute_jacobian(f, x):
    device = x.get_device()
    batch_size, traj_len, input_dim = f.shape

    # Flatten f and x
    f = f.view(batch_size, -1)

    # Initialize Jacobian matrix
    grad_dim = traj_len * input_dim

    jac = []
    # Iterate over each dimension
    for i in range(grad_dim):
        # Create gradient vector for the i-th output dimension
        grad_outputs = torch.zeros_like(f)
        grad_outputs[:, i] = 1.0

        # Compute gradients of f with respect to x
        gradients = torch.autograd.grad(
            outputs=f, inputs=x, grad_outputs=grad_outputs,
            create_graph=False, retain_graph=True, only_inputs=True
        )[0]
        
        # Fill in the Jacobian matrix
        jac.append(gradients.reshape(x.shape).squeeze(0))
    
    return torch.stack(jac).view(grad_dim, -1)