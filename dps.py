import yaml
import torch
import numpy as np
import image_tools
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

    def __init__(self, *, beta_schedule="linear", time_steps=1000, model_noise_type="gaussian", model_path="models/ffhq_10m.pt", step_size_factor = 0.1):
        self.N = time_steps
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        # Set up the model using dps_unet code (without modifications) and its corresponding model_config.yaml [2]
        self.model_noise_type = model_noise_type
        if model_path == "models/ffhq_10m.pt":
            self.model = self.load_model(model_path)
            self.learn_sigma = True
        else:
            raise NotImplementedError(f"Unknown model config for {model_path}")

        # *** The following parts of init func & its assertion checks referenced from Ho et al [3] and uses configs as specified by Chung et al [2] ***
        # Retrieve betas based on beta_schedule
        if beta_schedule.lower() == 'linear':
            # Linear schedule from Ho et al, extended to work for any number of diffusion steps by guided diffusion [4]
            scale = 1000 / self.N 
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            betas = torch.linspace(beta_start, beta_end, self.N, dtype=torch.float64).to(self.device)       
        else:
            # Currently only implemented linear because that's what the pretrained diffusion model I'm using has
            raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")
        self.betas = betas
        assert len(self.betas.shape) == 1
        assert len(self.betas) == self.N
        assert (self.betas > 0).all() and (self.betas <= 1).all()

        # Calculations for alphas and alpha-bars for ease of computation later
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, axis=0)
        self.alpha_bars_prev = torch.cat([torch.tensor([1.], dtype=torch.float64).to(self.device), self.alpha_bars[:-1]])
        assert len(self.alpha_bars) == len(self.alpha_bars_prev)
        assert self.alpha_bars_prev.shape == (self.N,)
        # *** This concludes referenced parts from Ho et al. ***
        
        # Calculations for q(x_t{t-1} | x_t), by def from original DDPM formulation
        self.one_minus_abars_t = 1. - self.alpha_bars
        self.one_minus_abars_tprev = 1. - self.alpha_bars_prev
        self.sqrt_abars_t = torch.sqrt(self.alpha_bars)
        self.sqrt_abars_tprev = torch.sqrt(self.alpha_bars_prev)
        self.ddpm_posterior_mean_xt_coeff = (torch.sqrt(self.alphas) * self.one_minus_abars_tprev) / self.one_minus_abars_t
        self.ddpm_posterior_mean_x0_coeff = (self.sqrt_abars_tprev * betas) / self.one_minus_abars_t
        self.ddpm_posterior_var = (self.one_minus_abars_tprev / self.one_minus_abars_t) * betas
        self.ddpm_log_posterior_var_clipped = torch.log(torch.cat((self.ddpm_posterior_var[1:2], self.ddpm_posterior_var[1:]))) # *** Referenced from Ho et al. ***

        # Initialize only hyperparameter of DPS, the step size
        # Instead of derived 1/self.ddpm_posterior_var (eq 16), we use formulation from foot notes #5 in original DPS paper
        # step_size = step_size_factor/||y − A(x0_hat)||, where step size factor is a constant between [0.01, 0.1]
        # Shown in Appendix C.4 (qualitatively through Figure 9) to produce more stable results 
        self.step_size_factor = step_size_factor
    
    def load_model(self, model_path="models/ffhq_10m.pt"):
        # Models must all end with ".pt" and their config files must be modelname_config.yaml
        with open(f"{model_path[:-3]}_config.yaml") as stream:
            try:
                params = yaml.safe_load(stream)
                return unet.create_model(**params).to(self.device)
            except yaml.YAMLError as exc:
                print(exc)

    def rescale(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)
    
    def learned_range_model_log_var(self, i, learned_sigma):
        min_log = self.ddpm_log_posterior_var_clipped[i]
        max_log = torch.log(self.betas)[i]
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (learned_sigma + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        return model_log_variance
    
    def gaussian_loss(self, res):
        return torch.sum(res**2)

    def poisson_loss(self, res, y):
        lam = torch.diag(1/2 * y.flatten())   # [***] how to do the y_j part? possibly a typo in the paper?
        # return res.T @ lam @ res
        return torch.matmul(torch.matmul(res.T, lam), res)

    def sample_conditional_posterior(self, y, A_operation, A_kwargs):
        """
        Performs Algorithm 1 from DPS paper [1], in which we can sample a posterior conditionally to solve inverse problems. Ie sample from p(x|y) given a noisy measurement y.

         - y: noisy and/or nonlinear measurement of something from p_data originally, goal is to produce a valid sample from the posterior p(x|y)
         - A_operation: the forward operator for the inverse problem. For eg, if the process is to mask out 30% of the image, this woudl be the function that handles the masking. 
        """
        
        # Intialize inverse problem, starting from x_N ~ pure noise
        # To compute gradient wrt x later, set requires_grad = True
        x = torch.randn_like(y, device=self.device, requires_grad=True)  # y.shape = (batch_size, height, width, channels)

        if not self.learn_sigma:
            raise NotImplementedError(f"All known models use learn_sigma = True, fixed variance is not yet unsupported.")
        print("y", torch.min(y), torch.max(y))
        pbar = tqdm(range(self.N-2, 0, -1))
        for i in pbar:
            print("x", torch.min(x), torch.max(x))
            # Since we have learned variance, it returns 3 channels for learned mean and other 3 for learned variance of the noise
            time_i = torch.tensor([i]).to(self.device)
            learned_eps = self.model(x, time_i)
            learned_eps_mean = learned_eps[:, :3, :, :]
            learned_eps_sigma = learned_eps[:, 3:, :, :]
            # [***] DOES SQRT GO HERE? DDPM AND DPS DIFFER BY A SQRT! 
            # A: yes! DPS is using s = grad(log(p(x_t))) = -1/(sqrt(1 - abar_t))*epislon
            x0_hat = 1/(self.sqrt_abars_t[i]) * (x - torch.sqrt(self.one_minus_abars_t[i])*learned_eps_mean) # x0_hat := E[x_0 | x_t] 
            print("xhat", torch.min(x0_hat), torch.max(x0_hat))
            
            # First do original DDPM posterior sampling for q(x_{t-1} | x_t)
            z = torch.randn_like(x, device=self.device)
            if i == 0:
                z = torch.zeros_like(x)

            # *** Referenced from Ho et al. ***
            # There are a few concepts that are specified in the diffusion_config.yaml file that were not explained by neither the DDPM paper nor the DPS paper, hence we needed to reference the original DDPM code for implementation details:
            
            # 1. clip_denoised: True
            x0_hat_clipped = x0_hat.clamp(-1, 1)  
            # 2. model_var_type: learned_range
            model_log_variance = self.learned_range_model_log_var(i, learned_eps_sigma)
            assert (model_log_variance.shape == x0_hat_clipped.shape == x.shape)

            # Finally, needed to see what they did with the learned sigmas, which also wasn't clear just the paper:
            sigma_i = torch.exp(0.5 * model_log_variance)
            ddpm_posterior = self.ddpm_posterior_mean_xt_coeff[i]*x + self.ddpm_posterior_mean_x0_coeff[i]*x0_hat_clipped + sigma_i*z
            # *** This concludes referenced parts from Ho et al. ***
            print("ddpm_post", torch.min(ddpm_posterior), torch.max(ddpm_posterior))
            
            
            # Likelihood calculations for DPS
            forward_x0_hat = A_operation(x0_hat_clipped, **A_kwargs)
            print("forward", torch.min(forward_x0_hat), torch.max(forward_x0_hat))
            res = torch.linalg.vector_norm(y - forward_x0_hat)
            if self.model_noise_type.lower() == 'gaussian':
                loss = res**2
            elif self.model_noise_type.lower() == 'poisson':
                loss = self.poisson_loss(res, y)
            else:
                # Currently only supporting Gaussian and Poisson from DPS paper formulations
                raise NotImplementedError(f"Unknown noise type: {self.model_noise_type}")
            grad_term = torch.autograd.grad(loss, x, create_graph=True)[0]
            print(f"Gradient term at step {i}: {grad_term}")
            true_step = self.step_size_factor/res # see footnotes #5 as mentioned in init function
            print("true_step", true_step)
            correction_term = -true_step*grad_term
            print("correction_term", torch.min(correction_term), torch.max(correction_term))
            x = ddpm_posterior + correction_term

            # Detach and clear the gradient to avoid accumulating graphs
            x = x.detach().clone().requires_grad_(True)
            torch.cuda.empty_cache()  # Clear unused memory, [***] IS THIS NEEDED?
            # if i % 50 == 0:
            #     image_tools.save_image(x, f"{A_kwargs['start_h']}_{A_kwargs['start_w']}/{i}.jpg")
        x = self.rescale(x)  # Rescale image back to 0 to 1 before saving
        return x

# [1]: Chung et al DPS paper: https://dps2022.github.io/diffusion-posterior-sampling-page/
# [2]: Chung et al DPS configs (for UNet and pretrained diffusion model): https://github.com/DPS2022/diffusion-posterior-sampling/tree/effbde7325b22ce8dc3e2c06c160c021e743a12d/configs 
# [3]: Ho et al DDPM gaussian-diffusion github code: https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
# [4]: Open AI guided-diffusion github code: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py 


