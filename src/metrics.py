"""Evaluation metrics for the audio branch + energy estimation helpers."""

import numpy as np
import torch


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = torch.sum(est * ref, dim=-1, keepdim=True)
    energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + eps
    proj = (dot / energy) * ref
    noise = est - proj
    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def snr_fn(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    noise = est - ref
    return 10 * torch.log10(
        (torch.sum(ref ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    )


def safe_mean(values) -> float:
    arr = np.array(values)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if len(finite) > 0 else 0.0


# ---------------------------------------------------------------------------
# Rigorous energy model (literature values)
# ---------------------------------------------------------------------------
# References (commonly cited in neuromorphic / SNN energy papers):
#   - Horowitz, M. "1.1 Computing's energy problem ...", ISSCC 2014
#     45 nm CMOS:  FP MAC ≈ 45 pJ,  32-bit ADD ≈ 0.9 pJ
#   - Merolla et al., TrueNorth; Davies et al., Loihi papers
#   - Rueckauer et al., "Conversion of continuous-valued deep networks to
#     efficient event-driven networks for image classification", Front. Neurosci.
#
# We report both:
#   1. Operation counts (MAC vs AC) derived from layer shapes
#   2. Estimated energy (pJ) = N_MAC * E_MAC + N_AC * E_AC
# using the Horowitz 45 nm numbers as a conservative digital baseline.
# Real neuromorphic chips (Loihi, TrueNorth) can be 1-2 orders of magnitude
# more efficient; those numbers can be substituted by the user.

E_MAC_PJ = 45.0   # pJ per multiply-accumulate (45 nm)
E_AC_PJ = 0.9     # pJ per accumulate / addition (45 nm)
E_SPIKE_PJ = 0.9  # approximate cost of a synaptic event / spike (same order)


def conv1d_macs(in_ch: int, out_ch: int, kernel: int, length: int) -> int:
    """MACs for a standard Conv1d (no groups/dilation adjustment needed for count)."""
    return in_ch * out_ch * kernel * length


def estimate_ann_energy(model, seq_len: int = 8000) -> dict:
    """Count MACs for a non-spiking 1-D residual / TCN network and convert
    to energy using literature pJ values.
    """
    total_macs = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            # output length ≈ input length for our padded convolutions
            total_macs += conv1d_macs(
                m.in_channels, m.out_channels, m.kernel_size[0], seq_len
            )
        elif isinstance(m, torch.nn.Linear):
            # time embedding projections act on batch, not sequence
            total_macs += m.in_features * m.out_features
    energy_pj = total_macs * E_MAC_PJ
    return {
        "macs": total_macs,
        "acs": 0,
        "energy_pj": energy_pj,
        "energy_uj": energy_pj / 1e6,
    }


def estimate_snn_energy(model, seq_len: int = 8000, spike_rate: float = 0.15,
                        num_steps: int = 8) -> dict:
    """Estimate energy for the spiking residual network.

    For each Conv1d that is driven by spikes we replace dense MACs by
    sparse ACs:  N_AC ≈ spike_rate * in_ch * out_ch * k * L * num_steps
    (the internal LIF unroll multiplies the number of synaptic events).

    Dense layers that are not spike-driven (time MLP, final projection
    in some designs) are still counted as MACs.
    """
    total_macs = 0
    total_acs = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv1d):
            dense = conv1d_macs(
                m.in_channels, m.out_channels, m.kernel_size[0], seq_len
            )
            # Heuristic: residual-block convolutions are spike-driven
            if "b1" in name or "b2" in name or "b3" in name or "blocks" in name:
                total_acs += int(dense * spike_rate * num_steps)
            else:
                # input / output projections often stay dense
                total_macs += dense
        elif isinstance(m, torch.nn.Linear):
            total_macs += m.in_features * m.out_features

    energy_pj = total_macs * E_MAC_PJ + total_acs * E_AC_PJ
    return {
        "macs": total_macs,
        "acs": total_acs,
        "energy_pj": energy_pj,
        "energy_uj": energy_pj / 1e6,
        "spike_rate_used": spike_rate,
        "num_steps_used": num_steps,
    }


def compare_energy(snn_stats: dict, ann_stats: dict) -> dict:
    """Return energy ratio and absolute savings."""
    ratio = ann_stats["energy_pj"] / max(snn_stats["energy_pj"], 1e-12)
    savings_pct = (1.0 - snn_stats["energy_pj"] / max(ann_stats["energy_pj"], 1e-12)) * 100
    return {
        "ann_energy_uj": ann_stats["energy_uj"],
        "snn_energy_uj": snn_stats["energy_uj"],
        "energy_ratio_ann_over_snn": ratio,
        "energy_savings_pct": savings_pct,
        "ann_macs": ann_stats["macs"],
        "snn_macs": snn_stats["macs"],
        "snn_acs": snn_stats["acs"],
    }
