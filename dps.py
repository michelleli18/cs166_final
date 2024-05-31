import torch
import numpy as np
from tqdm import tqdm
from autograd import grad
import dps_unet.unet as unet
import matplotlib.pyplot as plt

class DPS:
    """
    Performs Diffusion Posterior Sampling [1] given the beta_schedule, N, noise type, and path of the forward diffusion model along with a noisy and/or nonlinear measurement y to perform general inverse problem solving.

    Arguments:
     - beta_schedule: how the forward model chose its beta values
     - time_steps: = N in the original DDPM formulation, # of steps for noising process
     - model_noise_type: type of noise that the original model encoded, Ho et al offered Gaussian formulation, DPS provided possibilites for posterior sampling from both Gaussian and Poisson
     - model_path: path to the forward model
    """

    def __init__(self, *, beta_schedule="linear", time_steps=1000, model_noise_type="gaussian", model_path="models/ffhq_10m.pt"):
        self.N = time_steps

        # *** The following parts of init func & its assertion checks referenced from Ho et al [3] and uses configs as specified by Chung et al [2] ***
        # Retrieve betas based on beta_schedule
        if beta_schedule.lower() == 'linear':
            # Linear schedule from Ho et al, extended to work for any number of diffusion steps by guided diffusion [4]
            scale = 1000 / self.N 
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            betas = np.linspace(beta_start, beta_end, self.N, dtype=np.float64)
        else:
            # Currently only implemented linear because that's what the pretrained diffusion model I'm using has
            raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")
        self.betas = np.array(betas, dtype=np.float64)
        assert len(self.betas.shape) == 1
        assert len(self.betas) == self.N
        assert (self.betas > 0).all() and (self.betas <= 1).all()

        # Calculations for alphas and alpha-bars for ease of computation later
        self.alphas = 1. - self.betas
        self.alpha_bars = np.cumprod(self.alphas, axis=0)
        self.alpha_bars_prev = np.append(1., self.alpha_bars[:-1])
        assert len(self.alpha_bars == self.alpha_bars_prev)
        assert self.alpha_bars_prev.shape == (self.N,)
        # *** This concludes referenced parts from Ho et al. ***
        
        # Calculations for q(x_t{t-1} | x_t), by def from original DDPM formulation
        self.one_minus_abars_t = 1. - self.alpha_bars
        self.one_minus_abars_tprev = 1. - self.alpha_bars_prev
        self.sqrt_abars_t = np.sqrt(self.alpha_bars)
        self.sqrt_abars_tprev = np.sqrt(self.alpha_bars_prev)
        self.ddpm_posterior_mean_xt_coeff = (self.sqrt_abars_t * self.one_minus_abars_tprev) / self.one_minus_abars_t
        self.ddpm_posterior_mean_x0_coeff = (self.sqrt_abars_tprev * betas) / self.one_minus_abars_t
        self.ddpm_posterior_var = (self.one_minus_abars_tprev / self.one_minus_abars_t) * betas

        # Set up the model using dps_unet code (without modifications) and its corresponding model_config.yaml [2]
        self.model_noise_type = model_noise_type
        self.model = unet.create_model(image_size=256, num_channels=128, num_res_blocks=1, channel_mult="", learn_sigma=True, class_cond=False, use_checkpoint=False, attention_resolutions="16", num_heads=4, num_head_channels=64, num_heads_upsample=-1,use_scale_shift_norm=True, dropout=0, resblock_updown=True, use_fp16=False, use_new_attention_order=False, model_path=model_path) 
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            # self.model.load_state_dict(torch.load(model_path))
        else:
            self.device = 'cpu'
            # self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        # Initialize only hyperparameter of DPS, the step size
        # Instead of derived 1/self.ddpm_posterior_var (eq 16), we use formulation from foot notes #5 in original DPS paper
        # step_size = step_size_factor/||y − A(x0_hat)||, where step size factor is a constant between [0.01, 0.1]
        # Shown in Appendix C.4 (qualitatively through Figure 9) to produce more stable results 
        self.step_size_factor = 0.1

    def gaussian_loss(res):
        return np.sum(res**2)

    def poisson_loss(res, y):
        lam = np.diag(1/2*y) # [***] how to do the y_j part?
        return res.T @ lam @ res

    def sample_conditional_posterior(self, y):
        # y: noisy and/or nonlinear measurement of something from p_data originally, goal is to produce a valid sample from the posterior p(x|y)
        
        # Intialize inverse problem, starting from x_N
        # x = np.random.multivariate_normal(np.zeros(y.shape[-1]), np.identity(y.shape[-1]), size = y.shape[:3])
        x = np.random.normal(loc=0, scale=1, size=y.shape)
        x = np.transpose(x, (0, 3, 1, 2))       
        # x = np.transpose(x, (2, 0, 1))       
        x = torch.tensor(x).to(self.device)
        print(x.shape)

        # timestep_embs = unet.timestep_embedding(torch.tensor(list(range(self.N-1, 0, -1))).to(self.device), 2)

        pbar = tqdm(range(self.N-2, 0, -1))
        for i in pbar:
            # s = self.model(x, timestep_embs[i])
            s = self.model(x, torch.tensor([i]).to(self.device))
            print("YIAIII")
            # s = self.model(torch.zeros((1, 3, 256, 256)), torch.tensor([i]).to(self.device))
            x0_hat = 1/(self.alpha_bars[i]) * (x + (self.one_minus_abars_t[i])*s) # x0_hat := E[x_0 | x_t]
            
            # First do original DDPM posterior sampling for q(x_{t-1} | x_t)
            z = np.random.multivariate_normal(np.zeros(x.shape[-1]), np.identity(x.shape[-1]), size = x.shape)
            ddpm_posterior = self.ddpm_posterior_mean_xt_coeff*x + self.ddpm_posterior_mean_x0_coeff*x0_hat + np.sqrt(self.ddpm_posterior_var)*z
            
            # Likelihood calculations
            residual = y - self.model(x0_hat)
            if self.model_noise_type.lower() == 'gaussian':
                grad_term = grad(self.gaussian_loss(residual)) # [***] grad ok?
            if self.model_noise_type.lower() == 'poisson':
                grad_term = grad(self.poisson_loss(residual, y))
            else:
                # Currently only supporting Gaussian and Poisson from DPS paper formulations
                raise NotImplementedError(f"Unknown noise type: {self.model_noise_type}")
            true_step = self.step_size_factor/residual # see footnotes #5 as mentioned in init function
            correction_term = true_step*grad_term
            x = ddpm_posterior - correction_term
        return x # [***] return x_0 or x?


# [1]: Chung et al DPS paper: https://dps2022.github.io/diffusion-posterior-sampling-page/
# [2]: Chung et al DPS configs (for UNet and pretrained diffusion model): https://github.com/DPS2022/diffusion-posterior-sampling/tree/effbde7325b22ce8dc3e2c06c160c021e743a12d/configs 
# [3]: Ho et al DDPM gaussian-diffusion github code: https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
# [4]: Open AI guided-diffusion github code: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py 


