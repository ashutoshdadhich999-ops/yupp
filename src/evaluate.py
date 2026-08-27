"""
Evaluation utilities.

- Spiking vs non-spiking models are evaluated on the SAME corrupted
  batches (same RNG seed / same noise draw) for a fair paired comparison.
- Sparsity / timing measured on realistic (corrupted) inputs.
- Energy estimates derived from per-layer MAC / AC counts + literature pJ.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import snntorch as snn
from tqdm import tqdm

from src.diffusion_audio import PoissonDiffusion, gaussian_corrupt, bernoulli_corrupt
from src.metrics import (
    si_sdr, snr_fn, safe_mean,
    estimate_ann_energy, estimate_snn_energy, compare_energy,
)


# ---------------------------------------------------------------- image ----

def eval_img(model, name, test_loader, diff, timesteps, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    n_mse = d_mse = 0.0
    with torch.no_grad():
        for x, _ in tqdm(test_loader, desc=f"Eval {name}", leave=False):
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            xt = diff.q_sample(x, t, noise)
            pred = model(xt, t)

            a = torch.sqrt(diff.alpha_bar[t])[:, None, None, None]
            b = torch.sqrt(1 - diff.alpha_bar[t])[:, None, None, None]
            x0 = ((xt - b * pred) / a.clamp(1e-8)).clamp(0, 1)

            n_mse += F.mse_loss(xt, x).item()
            d_mse += F.mse_loss(x0, x).item()

    noisy = n_mse / len(test_loader)
    den = d_mse / len(test_loader)
    imp = ((noisy - den) / noisy) * 100
    print(f"[{name}] Noisy: {noisy:.5f} | Denoised: {den:.5f} | Imp: {imp:.2f}%")
    return noisy, den, imp


def eval_img_pair(model_a, model_b, name_a, name_b, test_loader, diff, timesteps,
                   device, seed=42):
    """Evaluate two image models on matched noise draws for a fair comparison."""
    res_a = eval_img(model_a, name_a, test_loader, diff, timesteps, device, seed=seed)
    res_b = eval_img(model_b, name_b, test_loader, diff, timesteps, device, seed=seed)
    return res_a, res_b


# ---------------------------------------------------------------- audio ----

def _corrupt(x0, t, T_audio, max_rate, device, mode="poisson"):
    if mode == "poisson":
        diff = PoissonDiffusion(T=T_audio, max_rate=max_rate, device=device)
        return diff.q_sample(x0, t)
    if mode == "gaussian":
        return gaussian_corrupt(x0, t, T_audio, device)
    if mode == "bernoulli":
        return bernoulli_corrupt(x0, t, T_audio, max_rate, device)
    raise ValueError(mode)


def evaluate_audio(model, name, test_loader, T_audio, max_rate, device,
                   seed=None, corrupt_mode="poisson"):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    mse_l, sdr_n, sdr_d, snr_n, snr_d = [], [], [], [], []
    with torch.no_grad():
        for x0 in tqdm(test_loader, desc=f"Eval {name}", leave=False):
            x0 = x0.to(device)
            t = torch.randint(T_audio // 3, T_audio, (x0.size(0),), device=device)
            noisy, _ = _corrupt(x0, t, T_audio, max_rate, device, corrupt_mode)
            pred = model(noisy, t)
            den = (noisy - pred).clamp(0, 1)

            mse_l.append(F.mse_loss(den, x0).item())
            sdr_n.extend(si_sdr(noisy, x0).cpu().tolist())
            sdr_d.extend(si_sdr(den, x0).cpu().tolist())
            snr_n.extend(snr_fn(noisy, x0).cpu().tolist())
            snr_d.extend(snr_fn(den, x0).cpu().tolist())

    res = {
        "MSE": safe_mean(mse_l),
        "SI-SDR Noisy": safe_mean(sdr_n),
        "SI-SDR Den": safe_mean(sdr_d),
        "SI-SDR Imp": safe_mean(sdr_d) - safe_mean(sdr_n),
        "SNR Noisy": safe_mean(snr_n),
        "SNR Den": safe_mean(snr_d),
        "SNR Imp": safe_mean(snr_d) - safe_mean(snr_n),
    }
    print(f"[{name}] MSE: {res['MSE']:.5f} | SI-SDR Imp: {res['SI-SDR Imp']:.2f} dB "
          f"| SNR Imp: {res['SNR Imp']:.2f} dB")
    return res


def evaluate_audio_pair(model_a, model_b, name_a, name_b, test_loader, T_audio,
                         max_rate, device, seed=42, corrupt_mode="poisson"):
    """Evaluate two audio models on matched noise draws for a fair comparison."""
    res_a = evaluate_audio(model_a, name_a, test_loader, T_audio, max_rate, device,
                           seed=seed, corrupt_mode=corrupt_mode)
    res_b = evaluate_audio(model_b, name_b, test_loader, T_audio, max_rate, device,
                           seed=seed, corrupt_mode=corrupt_mode)
    return res_a, res_b


# ------------------------------------------------------- sparsity / timing --

def measure_sparsity(model, test_loader, T_audio, max_rate, device, batches: int = 8):
    """Spike rate of a spiking audio model, measured on realistic (corrupted)
    input -- i.e. the same distribution the model sees during actual denoising."""
    model.eval()
    total_spikes = total_elements = 0.0
    spike_buffers = []

    def hook(module, inp, out):
        if isinstance(out, tuple):
            spike_buffers.append(out[0].detach())

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, snn.Leaky)]
    poisson_diff = PoissonDiffusion(T=T_audio, max_rate=max_rate, device=device)
    with torch.no_grad():
        for i, x0 in enumerate(test_loader):
            if i >= batches:
                break
            x0 = x0.to(device)
            t = torch.randint(0, T_audio, (x0.size(0),), device=device)
            noisy, _ = poisson_diff.q_sample(x0, t)
            spike_buffers.clear()
            _ = model(noisy, t)
            for spk in spike_buffers:
                total_spikes += spk.sum().item()
                total_elements += spk.numel()
    for h in hooks:
        h.remove()

    rate = total_spikes / (total_elements + 1e-8)
    return rate, 1 - rate


def measure_time(model_spiking, model_nonspiking, test_loader, T_audio, max_rate,
                  device, batches: int = 10, warmup: int = 3):
    """Latency of both audio models on the SAME realistic (corrupted) inputs."""
    poisson_diff = PoissonDiffusion(T=T_audio, max_rate=max_rate, device=device)

    def _time_one(model):
        model.eval()
        times = []
        with torch.no_grad():
            for i, x0 in enumerate(test_loader):
                if i >= warmup + batches:
                    break
                x0 = x0.to(device)
                t = torch.randint(0, T_audio, (x0.size(0),), device=device)
                noisy, _ = poisson_diff.q_sample(x0, t)

                if i < warmup:
                    _ = model(noisy, t)
                    continue
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(noisy, t)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000 / x0.size(0))
        return float(np.mean(times)), float(np.std(times))

    t_s, t_s_std = _time_one(model_spiking)
    t_ns, t_ns_std = _time_one(model_nonspiking)
    return (t_s, t_s_std), (t_ns, t_ns_std)


def run_energy_analysis(model_snn, model_ann, spike_rate: float, num_steps: int,
                        seq_len: int = 8000) -> dict:
    """Per-layer style energy comparison using literature pJ constants."""
    snn_stats = estimate_snn_energy(model_snn, seq_len=seq_len,
                                    spike_rate=spike_rate, num_steps=num_steps)
    ann_stats = estimate_ann_energy(model_ann, seq_len=seq_len)
    comparison = compare_energy(snn_stats, ann_stats)
    print("\n--- Energy Analysis (literature 45 nm CMOS model) ---")
    print(f"  ANN  MACs: {ann_stats['macs']:,}  ->  {ann_stats['energy_uj']:.3f} µJ")
    print(f"  SNN  MACs: {snn_stats['macs']:,}  ACs: {snn_stats['acs']:,}  "
          f"->  {snn_stats['energy_uj']:.3f} µJ  (spike_rate={spike_rate:.3f})")
    print(f"  Energy ratio (ANN/SNN): {comparison['energy_ratio_ann_over_snn']:.2f}x")
    print(f"  Estimated savings: {comparison['energy_savings_pct']:.1f}%")
    return comparison
