"""Training loops for the image and audio denoisers."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.diffusion_audio import PoissonDiffusion, gaussian_corrupt, bernoulli_corrupt


def train_img_model(model, name, train_loader, diff, timesteps, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"\nTraining {name}...")
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, _ in tqdm(train_loader, leave=False, desc=f"{name} Ep {ep + 1}"):
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            xt = diff.q_sample(x, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        print(f"Epoch {ep + 1}/{epochs}  Loss: {total / len(train_loader):.4f}")
    return model


def train_audio_model(model, name, train_loader, T_audio, max_rate, epochs, lr, device,
                      corrupt_mode: str = "poisson"):
    """Train an audio residual denoiser.

    corrupt_mode: "poisson" | "gaussian" | "bernoulli"
    Controls the forward process used for the ablation study.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    poisson_diff = PoissonDiffusion(T=T_audio, max_rate=max_rate, device=device)

    print(f"\nTraining {name} (corrupt={corrupt_mode})...")
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x0 in tqdm(train_loader, leave=False, desc=f"{name} Ep {ep + 1}"):
            x0 = x0.to(device)
            t = torch.randint(0, T_audio, (x0.size(0),), device=device)

            if corrupt_mode == "poisson":
                noisy, target = poisson_diff.q_sample(x0, t)
            elif corrupt_mode == "gaussian":
                noisy, target = gaussian_corrupt(x0, t, T_audio, device)
            elif corrupt_mode == "bernoulli":
                noisy, target = bernoulli_corrupt(x0, t, T_audio, max_rate, device)
            else:
                raise ValueError(f"Unknown corrupt_mode: {corrupt_mode}")

            pred = model(noisy, t)
            loss = F.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        print(f"Epoch {ep + 1}/{epochs}  Loss: {total / len(train_loader):.5f}")
    return model
