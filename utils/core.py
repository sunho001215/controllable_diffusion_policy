import torch

def compute_PMP(x_t, timestep, model, noise_scheduler):
    device = x_t.get_device()

    # Get alpha_t
    alpha_t = noise_scheduler.alphas_cumprod[timestep].to(device)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

    # Predict noise
    epsilon_theta = model(x_t, torch.tensor([timestep], dtype=torch.long).to(device))

    # Calcaulte PMP
    PMP = (x_t - sqrt_one_minus_alpha_t * epsilon_theta) / sqrt_alpha_t

    return PMP

def compute_jacobian(f, x):
    batch_size, _, _ = x.shape

    # Define a function that takes x and returns f
    def _func(x):
        # Ensure that f is computed with the correct shape, if needed
        return f(x).view(batch_size, -1)

    # Compute the Jacobian using torch's autograd functional method
    jacobian = torch.autograd.functional.jacobian(_func, x, vectorize=False)

    return jacobian

def compute_rank(matrix, eta=0.99):
    sum_of_squared_svdvals = torch.trace(torch.matmul(torch.transpose(matrix,0,1), matrix))

    top_r_sum_of_squared_svdvals = 0
    svdvals = torch.linalg.svdvals(matrix)
    for i in range(len(matrix)):
        top_r_sum_of_squared_svdvals += svdvals[i] ** 2
        if top_r_sum_of_squared_svdvals >= sum_of_squared_svdvals * (eta ** 2):
            break
    
    return i+1