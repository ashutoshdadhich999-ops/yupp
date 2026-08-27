"""
Proper Markov Poisson diffusion process for the audio branch.

Inspired by Poisson diffusion / count-data generative models
(e.g. thinning processes for point processes and discrete diffusion
on intensities). The forward process is defined as a Markov chain:

    q(x_t | x_{t-1}) = Poisson( x_{t-1} * (1 - beta_t) + beta_t * base_rate )

Equivalently, via sequential thinning + injection of independent
Poisson noise. Direct sampling q(x_t | x_0) is available via the
closed-form product of survival probabilities (gamma_bar).

This replaces the previous non-Markov "one-shot" corruption
x_t = f(x_0, t) and elevates the audio branch to a true diffusion
process, enabling the same theoretical framing used for the image
(DDPM) branch.
"""

import torch
import torch.nn.functional as F


class PoissonDiffusion:
    """Markov Poisson diffusion on continuous waveforms treated as
    intensity fields (after normalization to [0, 1]).

    Parameters
    ----------
    T : int
        Number of diffusion timesteps.
    max_rate : float
        Peak intensity scale (maps normalized waveform into Poisson rate).
    scale : float
        Multiplier that converts rate -> expected counts (higher = less
        relative Poisson shot noise).
    beta_start, beta_end : float
        Linear schedule for the thinning probability beta_t.
    jitter_std : float
        Small Gaussian observation noise layered on top (sensor jitter).
    """

    def __init__(
        self,
        T: int = 40,
        max_rate: float = 0.9,
        scale: float = 15.0,
        beta_start: float = 0.01,
        beta_end: float = 0.25,
        jitter_std: float = 0.04,
        device: str = "cpu",
    ):
        self.T = T
        self.max_rate = max_rate
        self.scale = scale
        self.jitter_std = jitter_std
        self.device = device

        # Linear beta schedule for thinning probability
        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        # Survival probability alpha_t = 1 - beta_t
        self.alpha = 1.0 - self.beta
        # Cumulative survival: gamma_bar_t = prod_{s=1..t} alpha_s
        self.gamma_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, return_target: bool = True):
        """Direct sample from q(x_t | x_0) using the closed-form
        cumulative thinning factor gamma_bar_t.

        x0 is assumed normalized to [0, 1].
        Returns (noisy, residual) where residual = noisy - x0
        (the quantity the residual denoiser is trained to predict).
        """
        # t shape: (B,)
        t = t.long().clamp(0, self.T - 1)
        gamma = self.gamma_bar[t].view(-1, *([1] * (x0.dim() - 1)))

        # Intensity after cumulative thinning
        rate = (x0 * gamma * self.max_rate).clamp(min=1e-4, max=self.max_rate)

        # Poisson draw (counts) then rescale back to intensity domain
        counts = torch.poisson(rate * self.scale)
        noisy = counts / self.scale

        # Additive Gaussian sensor jitter that grows as signal is destroyed.
        # gamma already has shape (B, 1, ...) so it broadcasts over the sequence.
        jitter_scale = (1.0 - gamma) * self.jitter_std
        noisy = (noisy + torch.randn_like(noisy) * jitter_scale).clamp(0.0, 1.0)

        if return_target:
            return noisy, noisy - x0
        return noisy

    def q_sample_step(self, x_prev: torch.Tensor, t: int):
        """Single Markov step q(x_t | x_{t-1}).

        Useful for ancestral sampling / reverse process experiments.
        t is a scalar timestep index (0-based).
        """
        beta_t = self.beta[t]
        # Thin previous intensity
        keep = torch.bernoulli(torch.full_like(x_prev, 1.0 - beta_t.item()))
        thinned = x_prev * keep
        # Inject independent Poisson noise scaled by beta
        inj_rate = (beta_t * self.max_rate * 0.5).clamp(min=1e-4)
        injection = torch.poisson(torch.full_like(x_prev, inj_rate.item()) * self.scale) / self.scale
        x_t = (thinned + injection).clamp(0.0, 1.0)
        return x_t

    def sample_trajectory(self, x0: torch.Tensor):
        """Generate the full forward trajectory x_0, x_1, ..., x_T
        by iterated Markov steps (for visualization / debugging).
        """
        traj = [x0]
        x = x0
        for t in range(self.T):
            x = self.q_sample_step(x, t)
            traj.append(x)
        return traj


# ---------------------------------------------------------------------------
# Alternative corruption processes used for the Poisson-vs-others ablation
# ---------------------------------------------------------------------------

def gaussian_corrupt(x0: torch.Tensor, t: torch.Tensor, T: int, device: str,
                     noise_scale: float = 0.35):
    """Standard Gaussian additive noise whose variance grows with t.
    Matched dynamic range for fair comparison against Poisson diffusion.
    """
    t = t.view(-1, 1).float().to(device)
    sigma = noise_scale * (t / max(T - 1, 1)).clamp(0.05, 1.0)
    noise = torch.randn_like(x0) * sigma
    noisy = (x0 + noise).clamp(0.0, 1.0)
    return noisy, noisy - x0


def bernoulli_corrupt(x0: torch.Tensor, t: torch.Tensor, T: int, max_rate: float,
                      device: str, scale: float = 15.0):
    """Bernoulli spike encoding: each sample is independently kept with
    probability proportional to intensity * survival, producing binary
    spike trains (then rate-normalized). Useful biological baseline.
    """
    t = t.view(-1, 1).float().to(device)
    gamma = torch.exp(-0.08 * t * 6 / T)  # similar decay envelope
    p = (x0 * gamma * max_rate).clamp(1e-4, 0.99)
    # Multiple Bernoulli trials averaged to keep continuous-valued output
    spikes = 0.0
    n_trials = 8
    for _ in range(n_trials):
        spikes = spikes + torch.bernoulli(p)
    noisy = (spikes / n_trials).clamp(0.0, 1.0)
    return noisy, noisy - x0


def poison(x0, t, T_audio, max_rate, device, scale=15.0, decay=0.06, jitter_std=0.05):
    """Backward-compatible wrapper used by older call sites.

    Internally constructs a PoissonDiffusion on the fly so that existing
    train / eval code continues to work without immediate rewrite.
    Prefer constructing PoissonDiffusion once and calling .q_sample.
    """
    diff = PoissonDiffusion(
        T=T_audio, max_rate=max_rate, scale=scale,
        jitter_std=jitter_std, device=device,
    )
    return diff.q_sample(x0, t, return_target=True)
