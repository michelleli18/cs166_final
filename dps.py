import torch
import operator 
import numpy as np
from autograd import grad
from itertools import accumulate
from tqdm.auto import tqdm


def gaussian_loss(res):
    return np.sum(res**2)

def poisson_loss(res):
    lam = np.diag(1/2*y) # [***] how to do the y_j part?
    return res.T @ lam @ res

def denoise(N, y, step_sizes, sigmas, model, noise_type='Gaussian'):
    pbar = tqdm(list(range(N-1, 1))[::-1])
    for i in pbar:
        s = model(x)
        abar = get_alpha_bar(i, alphas) # [***] where to get these alphas
        x_0 = 1/(abar) * (x + (1-abar)*s)
        z = np.random.multivariate_normal([0, 0, 0], np.identity(3))
        abar_prev = get_alpha_bar(i-1, alphas)
        x_next = ((np.sqrt(alphas[i])(1 - abar_prev))/(1 - abar))*x \
        + ((np.sqrt(abar_prev)*beta[i])/(1 - abar_prev))*x_0 \
        + sigmas[i]*z
        res = y - model(x_0)
        # currently only supporting Gaussian and Poisson from DPS paper formulations
        if noise_type == 'Gaussian':
            grad_term = grad(gaussian_loss(res)) # [***] grad ok?
        if noise_type == 'Poisson':
            grad_term = grad(poisson_loss(res))
        else:
            raise ValueError("unsupported noise type")
        true_step = step_sizes[i]/res # footnotes #5, steps_sizes[i] is constant
        rho = 1/(sigmas[i]**2) # bottom of P5
        correction_term = true_step*rho*grad_term
        x = x_next - correction_term
    return x # [***] return x_0 or x?


model_path = "ffhq_10m.pt"
model = torch.load(model_path, map_location=torch.device('cpu'))

N = 1000 # find a valid N val
y = np.random.normal(size=(200, 200)) # [***] how big images?
x_N = np.random.multivariate_normal([0, 0, 0], np.identity(3)
, size=(5, 5, 3))  # [***] what is dim of cov, is it 3 for color images?
step_sizes = np.full((N,), 0.01)  # [***] hold constant? they said [0.01, 0.1] is best


# [***] how to get these from the forward model?
# sigmas = ??
# betas = ??
# alphas = ??



# print("Names of Tensors Stored in the Model:")
# for key in model.keys():
#     print(key)


# [1]: Chung et al DPS paper: 
# [2]: Ho et al DDPM gaussian-diffusion github code: https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
# [3]: Open AI guided-diffusion github code: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py 