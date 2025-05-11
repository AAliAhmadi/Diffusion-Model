import torch
import torch.nn.functional as F

class DiffusionScheduler:
    def __init__(self, timesteps, start_beta=0.0001, end_beta=0.02):
        self.timesteps = timesteps
        self.schedule = self.prepare_noise_schedule(timesteps, start_beta, end_beta)

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def prepare_noise_schedule(self, T, start_beta=0.0001, end_beta=0.02):
        betas = self.linear_beta_schedule(T, start=start_beta, end=end_beta)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        return {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_recip_alphas": torch.sqrt(1. / alphas),
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1. - alphas_cumprod),
            "posterior_variance": betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        }

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.schedule["sqrt_alphas_cumprod"][t]
        sqrt_one_minus_alphas_cumprod_t = self.schedule["sqrt_one_minus_alphas_cumprod"][t]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def sample(self, model, shape, device, denoise_fn=None):
        x = torch.randn(shape, device=device)  # Start from Gaussian noise
    
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            beta_t = self.schedule["betas"][t]
            sqrt_one_minus_alphas_cumprod_t = self.schedule["sqrt_one_minus_alphas_cumprod"][t]
            sqrt_recip_alphas_t = self.schedule["sqrt_recip_alphas"][t]
    
            # Predict noise
            eps_theta = model(x, t_batch)
    
            # Compute x0 prediction
            x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * eps_theta) / sqrt_recip_alphas_t
    
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
    
            posterior_var = self.schedule["posterior_variance"][t]
            mean = (
                self.schedule["sqrt_alphas_cumprod"][t] * x0_pred
                + sqrt_one_minus_alphas_cumprod_t * noise
            )
    
            x = mean + torch.sqrt(posterior_var) * noise
    
        return x

