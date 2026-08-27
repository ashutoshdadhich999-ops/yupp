"""Plotting utilities. Figures are saved to disk (script-safe)."""

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.diffusion_audio import PoissonDiffusion

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 11


def plot_image_samples(model, test_loader, diff, timesteps, device, out_dir, n=8):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x[:n].to(device)
        t = torch.full((n,), timesteps // 2, device=device)
        noise = torch.randn_like(x)
        xt = diff.q_sample(x, t, noise)
        pred = model(xt, t)
        a = torch.sqrt(diff.alpha_bar[t])[:, None, None, None]
        b = torch.sqrt(1 - diff.alpha_bar[t])[:, None, None, None]
        x0_pred = ((xt - b * pred) / a.clamp(1e-8)).clamp(0, 1)

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(xt[i, 0].cpu(), cmap="gray"); axes[1, i].axis("off")
        axes[2, i].imshow(x0_pred[i, 0].cpu(), cmap="gray"); axes[2, i].axis("off")
    axes[0, 0].set_title("Clean", fontsize=10)
    axes[1, 0].set_title("Noisy", fontsize=10)
    axes[2, 0].set_title("Denoised (Spiking)", fontsize=10)
    plt.suptitle("Image Denoising Examples (Spiking Model)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "image_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_audio_waveforms(model, test_loader, T_audio, max_rate, device, out_dir, n=4):
    poisson_diff = PoissonDiffusion(T=T_audio, max_rate=max_rate, device=device)
    model.eval()
    with torch.no_grad():
        x0 = next(iter(test_loader))[:n].to(device)
        t = torch.full((n,), int(T_audio * 0.7), device=device)
        noisy, _ = poisson_diff.q_sample(x0, t)
        pred = model(noisy, t)
        den = (noisy - pred).clamp(0, 1)

    fig, axes = plt.subplots(n, 1, figsize=(14, 2 * n), sharex=True)
    for i in range(n):
        axes[i].plot(x0[i].detach().cpu().numpy(), label="Clean", alpha=0.8, linewidth=1.2)
        axes[i].plot(noisy[i].detach().cpu().numpy(), label="Noisy", alpha=0.6, linewidth=1)
        axes[i].plot(den[i].detach().cpu().numpy(), label="Denoised (Spiking)", alpha=0.9, linewidth=1.2)
        axes[i].set_ylabel(f"Sample {i + 1}")
        axes[i].legend(loc="upper right", fontsize=9)
        axes[i].set_ylim(-0.05, 1.05)
    axes[-1].set_xlabel("Time steps")
    plt.suptitle("Audio Denoising Examples (Spiking Model)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "audio_waveforms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_temporal_conditioning(model, test_loader, T_audio, max_rate, device, out_dir,
                               t_values=None, n_show=2):
    """Qualitative proof that the network actually uses the timestep condition.

    For a fixed clean waveform we corrupt it at several t and show the
    model's residual prediction / reconstruction. Distinct behaviour across
    low / medium / high noise regimes indicates that temporal conditioning
    was learned.
    """
    if t_values is None:
        t_values = [0, 5, 10, 20, 30, min(39, T_audio - 1)]

    poisson_diff = PoissonDiffusion(T=T_audio, max_rate=max_rate, device=device)
    model.eval()
    n_t = len(t_values)
    fig, axes = plt.subplots(n_show, n_t, figsize=(3.2 * n_t, 2.8 * n_show), sharey=True)
    if n_show == 1:
        axes = axes[None, :]

    with torch.no_grad():
        x0 = next(iter(test_loader))[:n_show].to(device)
        for row in range(n_show):
            for col, t_val in enumerate(t_values):
                t = torch.full((1,), t_val, device=device)
                noisy, _ = poisson_diff.q_sample(x0[row:row + 1], t)
                pred = model(noisy, t)
                den = (noisy - pred).clamp(0, 1)

                ax = axes[row, col]
                ax.plot(x0[row].detach().cpu().numpy(), color="C0", alpha=0.5,
                        linewidth=0.9, label="clean")
                ax.plot(noisy[0].detach().cpu().numpy(), color="C1", alpha=0.55,
                        linewidth=0.8, label="noisy")
                ax.plot(den[0].detach().cpu().numpy(), color="C2", alpha=0.9,
                        linewidth=1.0, label="denoised")
                ax.set_ylim(-0.05, 1.05)
                if row == 0:
                    ax.set_title(f"t = {t_val}", fontsize=11)
                if col == 0:
                    ax.set_ylabel(f"Sample {row + 1}")
                if row == 0 and col == n_t - 1:
                    ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Temporal Conditioning: Denoising Behaviour vs. Noise Level t",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "temporal_conditioning.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_corruption_ablation(results: dict, out_dir):
    """Bar chart comparing Poisson / Gaussian / Bernoulli forward processes."""
    modes = list(results.keys())
    mse_vals = [results[m]["MSE"] for m in modes]
    sdr_vals = [results[m]["SI-SDR Imp"] for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    axes[0].bar(modes, mse_vals, color=colors[: len(modes)])
    axes[0].set_title("Denoised MSE (lower better)")
    axes[0].set_ylabel("MSE")

    axes[1].bar(modes, sdr_vals, color=colors[: len(modes)])
    axes[1].set_title("SI-SDR Improvement (higher better)")
    axes[1].set_ylabel("dB")

    plt.suptitle("Forward Process Ablation: Poisson vs Gaussian vs Bernoulli",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    path = os.path.join(out_dir, "corruption_ablation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison_bars(img_s_imp, img_ns_imp, audio_s_sdr_imp, audio_ns_sdr_imp,
                          sparsity, out_dir, energy_savings=None):
    n_plots = 4 if energy_savings is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(4.2 * n_plots, 4.5))

    axes[0].bar(["Spiking", "Non-Spiking"], [img_s_imp, img_ns_imp],
                color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Image: MSE Improvement (%)")
    axes[0].set_ylabel("Improvement %")
    axes[0].set_ylim(0, 100)

    axes[1].bar(["Spiking", "Non-Spiking"], [audio_s_sdr_imp, audio_ns_sdr_imp],
                color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Audio: SI-SDR Improvement (dB)")
    axes[1].set_ylabel("dB")

    axes[2].bar(["Spiking", "Non-Spiking"], [sparsity, 0.0],
                color=["#4C72B0", "#DD8452"])
    axes[2].set_title("Sparsity")
    axes[2].set_ylabel("Sparsity (1 - spike rate)")
    axes[2].set_ylim(0, 1)

    if energy_savings is not None:
        axes[3].bar(["Spiking", "Non-Spiking"],
                    [energy_savings, 0.0],
                    color=["#4C72B0", "#DD8452"])
        axes[3].set_title("Est. Energy Savings (%)")
        axes[3].set_ylabel("% relative to ANN")
        axes[3].set_ylim(0, 100)

    plt.suptitle("Spiking vs Non-Spiking Comparison", fontsize=14, y=1.03)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
