"""
Spiking vs Non-Spiking Residual Denoising -- main experiment script.

Upgrades vs. original:
  1. Audio branch uses a proper Markov Poisson diffusion process
     q(x_t | x_{t-1}) with closed-form q(x_t | x_0).
  2. Rigorous energy analysis (MAC/AC counts × literature pJ values).
  3. Forward-process ablation: Poisson vs Gaussian vs Bernoulli.
  4. Temporal-conditioning visualization (denoising at multiple t).
  5. Stronger ANN baseline (dilated residual TCN) so SNN results
     cannot be dismissed as "weak baseline".

Usage:
    python main.py                     # run everything with defaults
    python main.py --skip-audio        # image branch only
    python main.py --skip-image        # audio branch only
    python main.py --epochs-img 5 --epochs-audio 5   # quick smoke test
    python main.py --skip-ablation     # skip Poisson/Gaussian/Bernoulli study
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import torchaudio

from src.diffusion_image import Diffusion
from src.models_image import StrongImgNet, NonSpikeImgNet
from src.models_audio import StrongAudioNet, NonSpikeAudioNet, StrongANNAudioNet
from src.datasets import AudioDS
from src.train import train_img_model, train_audio_model
from src.evaluate import (
    eval_img_pair, evaluate_audio_pair, evaluate_audio,
    measure_sparsity, measure_time, run_energy_analysis,
)
from src.visualize import (
    plot_image_samples, plot_audio_waveforms, plot_comparison_bars,
    plot_temporal_conditioning, plot_corruption_ablation,
)


def parse_args():
    p = argparse.ArgumentParser(description="Spiking vs Non-Spiking Residual Denoising")
    p.add_argument("--skip-image", action="store_true")
    p.add_argument("--skip-audio", action="store_true")
    p.add_argument("--skip-ablation", action="store_true",
                   help="Skip Poisson/Gaussian/Bernoulli forward-process ablation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs")

    # Image branch
    p.add_argument("--batch-size-img", type=int, default=64)
    p.add_argument("--epochs-img", type=int, default=12)
    p.add_argument("--timesteps-img", type=int, default=20)
    p.add_argument("--num-steps-img", type=int, default=5)
    p.add_argument("--base-channels-img", type=int, default=32)
    p.add_argument("--lr-img", type=float, default=2e-4)

    # Audio branch
    p.add_argument("--audio-len", type=int, default=8000)
    p.add_argument("--audio-sr", type=int, default=16000)
    p.add_argument("--timesteps-audio", type=int, default=40)
    p.add_argument("--num-steps-audio", type=int, default=8)
    p.add_argument("--epochs-audio", type=int, default=20)
    p.add_argument("--batch-size-audio", type=int, default=32)
    p.add_argument("--max-rate-audio", type=float, default=0.9)
    p.add_argument("--lr-audio", type=float, default=3e-4)
    p.add_argument("--audio-subset-size", type=int, default=6000)
    p.add_argument("--ablation-epochs", type=int, default=8,
                   help="Epochs for the shorter corruption-mode ablation runs")

    return p.parse_args()


def run_image_branch(args, device):
    print("\n" + "=" * 70)
    print("PART 1: MNIST Image Denoising")
    print("=" * 70)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size_img, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size_img, shuffle=False,
                              num_workers=2, pin_memory=True)

    diff = Diffusion(T=args.timesteps_img, device=device)

    model_img = StrongImgNet(base_channels=args.base_channels_img,
                              num_steps=args.num_steps_img).to(device)
    model_ns_img = NonSpikeImgNet(base_channels=args.base_channels_img).to(device)

    model_img = train_img_model(model_img, "Spiking Image", train_loader, diff,
                                 args.timesteps_img, args.epochs_img, args.lr_img, device)
    model_ns_img = train_img_model(model_ns_img, "Non-Spiking Image", train_loader, diff,
                                    args.timesteps_img, args.epochs_img, args.lr_img, device)

    print("\n--- Image Results (paired, matched noise draws) ---")
    (img_s_noisy, img_s_den, img_s_imp), (img_ns_noisy, img_ns_den, img_ns_imp) = eval_img_pair(
        model_img, model_ns_img, "Spiking Image", "Non-Spiking Image",
        test_loader, diff, args.timesteps_img, device, seed=args.seed,
    )

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_image_samples(model_img, test_loader, diff, args.timesteps_img, device, fig_dir)

    return {
        "img_s_noisy": img_s_noisy, "img_s_den": img_s_den, "img_s_imp": img_s_imp,
        "img_ns_noisy": img_ns_noisy, "img_ns_den": img_ns_den, "img_ns_imp": img_ns_imp,
    }


def run_audio_branch(args, device):
    print("\n" + "=" * 70)
    print("PART 2: Audio Denoising (SpeechCommands) — Markov Poisson Diffusion")
    print("=" * 70)

    os.makedirs("./data", exist_ok=True)
    base = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True)
    subset = Subset(base, range(min(args.audio_subset_size, len(base))))
    tr_size = int(0.85 * len(subset))
    tr_sub, te_sub = random_split(subset, [tr_size, len(subset) - tr_size])

    train_loader = DataLoader(AudioDS(tr_sub, args.audio_len, args.audio_sr),
                               batch_size=args.batch_size_audio, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(AudioDS(te_sub, args.audio_len, args.audio_sr),
                              batch_size=args.batch_size_audio, shuffle=False,
                              num_workers=2, pin_memory=True)

    # --- Primary models: SNN + strong ANN baseline (dilated TCN) ---
    model_a = StrongAudioNet(num_steps=args.num_steps_audio,
                              T_audio=args.timesteps_audio).to(device)
    model_strong_ann = StrongANNAudioNet(T_audio=args.timesteps_audio).to(device)
    # Topology-matched non-spiking residual (kept for completeness)
    model_ns_a = NonSpikeAudioNet(T_audio=args.timesteps_audio).to(device)

    model_a = train_audio_model(model_a, "Spiking Audio", train_loader,
                                 args.timesteps_audio, args.max_rate_audio,
                                 args.epochs_audio, args.lr_audio, device,
                                 corrupt_mode="poisson")
    model_strong_ann = train_audio_model(
        model_strong_ann, "Strong ANN (Dilated TCN)", train_loader,
        args.timesteps_audio, args.max_rate_audio,
        args.epochs_audio, args.lr_audio, device, corrupt_mode="poisson",
    )
    model_ns_a = train_audio_model(model_ns_a, "Matched Non-Spiking Audio", train_loader,
                                    args.timesteps_audio, args.max_rate_audio,
                                    args.epochs_audio, args.lr_audio, device,
                                    corrupt_mode="poisson")

    print("\n--- Audio Results vs Strong ANN baseline (paired) ---")
    res_s, res_strong = evaluate_audio_pair(
        model_a, model_strong_ann, "Spiking Audio", "Strong ANN (Dilated TCN)",
        test_loader, args.timesteps_audio, args.max_rate_audio, device, seed=args.seed,
    )
    print("\n--- Audio Results vs topology-matched Non-Spiking ---")
    _, res_ns = evaluate_audio_pair(
        model_a, model_ns_a, "Spiking Audio", "Matched Non-Spiking",
        test_loader, args.timesteps_audio, args.max_rate_audio, device, seed=args.seed,
    )

    spike_rate, sparsity = measure_sparsity(model_a, test_loader, args.timesteps_audio,
                                             args.max_rate_audio, device)
    (t_s, t_s_std), (t_ns, t_ns_std) = measure_time(
        model_a, model_strong_ann, test_loader,
        args.timesteps_audio, args.max_rate_audio, device,
    )

    # Rigorous energy analysis
    energy = run_energy_analysis(
        model_a, model_strong_ann, spike_rate=spike_rate,
        num_steps=args.num_steps_audio, seq_len=args.audio_len,
    )

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_audio_waveforms(model_a, test_loader, args.timesteps_audio,
                          args.max_rate_audio, device, fig_dir)
    plot_temporal_conditioning(model_a, test_loader, args.timesteps_audio,
                                args.max_rate_audio, device, fig_dir)

    return {
        "res_s": res_s,
        "res_strong": res_strong,
        "res_ns": res_ns,
        "spike_rate": spike_rate,
        "sparsity": sparsity,
        "t_s": t_s, "t_s_std": t_s_std,
        "t_ns": t_ns, "t_ns_std": t_ns_std,
        "energy": energy,
        "model_a": model_a,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }


def run_corruption_ablation(args, device, train_loader, test_loader):
    """Ablation: train the *same* SNN architecture under three forward processes
    (Poisson / Gaussian / Bernoulli) and compare final denoising quality.

    Justifies the choice of Poisson encoding. If Poisson wins we can claim
    that biological spike statistics better preserve information; a negative
    result is still scientifically valuable.
    """
    print("\n" + "=" * 70)
    print("PART 3: Forward-Process Ablation (Poisson / Gaussian / Bernoulli)")
    print("=" * 70)

    results = {}
    for mode in ["poisson", "gaussian", "bernoulli"]:
        model = StrongAudioNet(num_steps=args.num_steps_audio,
                                T_audio=args.timesteps_audio).to(device)
        model = train_audio_model(
            model, f"SNN ({mode})", train_loader,
            args.timesteps_audio, args.max_rate_audio,
            args.ablation_epochs, args.lr_audio, device,
            corrupt_mode=mode,
        )
        res = evaluate_audio(
            model, f"SNN ({mode})", test_loader,
            args.timesteps_audio, args.max_rate_audio, device,
            seed=args.seed, corrupt_mode=mode,
        )
        results[mode] = res

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_corruption_ablation(results, fig_dir)

    print("\n--- Ablation Summary ---")
    print(f"{'Mode':<12} {'MSE':>10} {'SI-SDR Imp':>12}")
    print("-" * 36)
    for mode, res in results.items():
        print(f"{mode:<12} {res['MSE']:>10.5f} {res['SI-SDR Imp']:>12.2f}")
    return results


def print_final_tables(img_results, audio_results, ablation_results=None):
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    if img_results:
        print("\n[IMAGE]")
        print(f"{'Model':<25} {'Noisy MSE':>12} {'Denoised MSE':>14} {'Improvement':>12}")
        print("-" * 65)
        print(f"{'Spiking':<25} {img_results['img_s_noisy']:>12.5f} "
              f"{img_results['img_s_den']:>14.5f} {img_results['img_s_imp']:>11.2f}%")
        print(f"{'Non-Spiking':<25} {img_results['img_ns_noisy']:>12.5f} "
              f"{img_results['img_ns_den']:>14.5f} {img_results['img_ns_imp']:>11.2f}%")

    if audio_results:
        res_s = audio_results["res_s"]
        res_strong = audio_results["res_strong"]
        res_ns = audio_results["res_ns"]
        print("\n[AUDIO]  (primary comparison vs Strong ANN / Dilated TCN)")
        print(f"{'Metric':<22} {'Spiking':>12} {'Strong ANN':>14} {'Matched NS':>14}")
        print("-" * 64)
        print(f"{'MSE (Denoised)':<22} {res_s['MSE']:>12.5f} "
              f"{res_strong['MSE']:>14.5f} {res_ns['MSE']:>14.5f}")
        print(f"{'SI-SDR Imp (dB)':<22} {res_s['SI-SDR Imp']:>12.2f} "
              f"{res_strong['SI-SDR Imp']:>14.2f} {res_ns['SI-SDR Imp']:>14.2f}")
        print(f"{'SNR Imp (dB)':<22} {res_s['SNR Imp']:>12.2f} "
              f"{res_strong['SNR Imp']:>14.2f} {res_ns['SNR Imp']:>14.2f}")
        print(f"{'Spike Rate':<22} {audio_results['spike_rate']:>12.4f} "
              f"{'1.0000':>14} {'1.0000':>14}")
        print(f"{'Sparsity':<22} {audio_results['sparsity']:>12.4f} "
              f"{'0.0000':>14} {'0.0000':>14}")
        print(f"{'Time (ms/sample)':<22} "
              f"{audio_results['t_s']:>9.3f}\u00b1{audio_results['t_s_std']:.2f} "
              f"{audio_results['t_ns']:>10.3f}\u00b1{audio_results['t_ns_std']:.2f}")

        if "energy" in audio_results:
            e = audio_results["energy"]
            print(f"\n[ENERGY]  (45 nm CMOS literature model)")
            print(f"  ANN energy : {e['ann_energy_uj']:.3f} µJ")
            print(f"  SNN energy : {e['snn_energy_uj']:.3f} µJ")
            print(f"  Ratio ANN/SNN : {e['energy_ratio_ann_over_snn']:.2f}x")
            print(f"  Savings : {e['energy_savings_pct']:.1f}%")

    if ablation_results:
        print("\n[FORWARD-PROCESS ABLATION]")
        print(f"{'Mode':<12} {'MSE':>10} {'SI-SDR Imp':>12}")
        print("-" * 36)
        for mode, res in ablation_results.items():
            print(f"{mode:<12} {res['MSE']:>10.5f} {res['SI-SDR Imp']:>12.2f}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs(args.out_dir, exist_ok=True)

    img_results = run_image_branch(args, device) if not args.skip_image else None
    audio_results = run_audio_branch(args, device) if not args.skip_audio else None

    ablation_results = None
    if audio_results is not None and not args.skip_ablation:
        ablation_results = run_corruption_ablation(
            args, device,
            audio_results["train_loader"],
            audio_results["test_loader"],
        )

    print_final_tables(img_results, audio_results, ablation_results)

    if img_results and audio_results:
        fig_dir = os.path.join(args.out_dir, "figures")
        energy_sav = None
        if "energy" in audio_results:
            energy_sav = audio_results["energy"]["energy_savings_pct"]
        plot_comparison_bars(
            img_results["img_s_imp"], img_results["img_ns_imp"],
            audio_results["res_s"]["SI-SDR Imp"],
            audio_results["res_strong"]["SI-SDR Imp"],
            audio_results["sparsity"], fig_dir,
            energy_savings=energy_sav,
        )

    print("\n" + "=" * 70)
    print(f"ALL DONE. Figures saved under: {os.path.join(args.out_dir, 'figures')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
