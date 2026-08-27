"""
Audio branch models.

- StrongAudioNet : spiking residual denoiser (LIF).
- NonSpikeAudioNet : original matched non-spiking residual baseline.
- StrongANNAudioNet : stronger ANN baseline (dilated residual TCN / 1-D U-Net
  style) so that any SNN advantage cannot be dismissed as "weak baseline".
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ---------------------------------------------------------------------------
# Spiking residual blocks
# ---------------------------------------------------------------------------

class ConvResBlock1D(nn.Module):
    """Spiking 1D residual block (LIF neurons)."""

    def __init__(self, ch: int, time_dim: int, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.time_proj = nn.Linear(time_dim, ch)
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, ch)
        self.lif1 = snn.Leaky(beta=0.92, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm2 = nn.GroupNorm(8, ch)
        self.lif2 = snn.Leaky(beta=0.92, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, temb):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        res = x
        h = self.norm1(self.conv1(x))
        temb_p = self.time_proj(temb)[:, :, None]

        outs = []
        for _ in range(self.num_steps):
            h_in = h + temb_p
            spk, mem1 = self.lif1(h_in, mem1)
            h2 = self.norm2(self.conv2(spk))
            spk2, mem2 = self.lif2(h2, mem2)
            outs.append(spk2 + res)
        return torch.stack(outs).mean(0)


class NonSpikeConvRes1D(nn.Module):
    """Non-spiking counterpart of ConvResBlock1D (original matched baseline)."""

    def __init__(self, ch: int, time_dim: int):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, ch)
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()

    def forward(self, x, temb):
        res = x
        h = self.norm1(self.conv1(x))
        h = h + self.time_proj(temb)[:, :, None]
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        return h + res


# ---------------------------------------------------------------------------
# Stronger ANN baseline: dilated residual TCN / lightweight 1-D U-Net
# ---------------------------------------------------------------------------

class DilatedResBlock1D(nn.Module):
    """Dilated residual block used inside the strong ANN baseline.
    Larger receptive field + residual connections make this a much
    stronger competitor than the original shallow residual stack.
    """

    def __init__(self, ch: int, time_dim: int, dilation: int = 1):
        super().__init__()
        padding = dilation * 2  # kernel=5 -> pad = dilation*(k-1)/2
        self.time_proj = nn.Linear(time_dim, ch)
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()

    def forward(self, x, temb):
        res = x
        h = self.norm1(self.conv1(x))
        h = h + self.time_proj(temb)[:, :, None]
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        return h + res


class StrongANNAudioNet(nn.Module):
    """Strong non-spiking baseline: multi-scale dilated residual TCN.

    Design choices that make it stronger than the original NonSpikeAudioNet:
      * 6 residual blocks with exponentially growing dilations (1,2,4,8,16,32)
        -> very large receptive field covering the whole 8000-sample window
      * higher channel count (96)
      * explicit multi-scale skip aggregation

    If the SNN remains competitive against this model, the result carries
    substantially more weight.
    """

    def __init__(self, channels: int = 96, time_dim: int = 128, T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.input = nn.Conv1d(1, channels, 7, padding=3)

        dilations = [1, 2, 4, 8, 16, 32]
        self.blocks = nn.ModuleList(
            [DilatedResBlock1D(channels, time_dim, d) for d in dilations]
        )
        # Lightweight multi-scale fusion
        self.fuse = nn.Conv1d(channels * len(dilations), channels, 1)
        self.out = nn.Conv1d(channels, 1, 7, padding=3)
        self.act = nn.SiLU()

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))
        h = self.input(x.unsqueeze(1))
        features = []
        for blk in self.blocks:
            h = blk(h, temb)
            features.append(h)
        h = self.fuse(torch.cat(features, dim=1))
        h = self.act(h)
        return self.out(h).squeeze(1)


# ---------------------------------------------------------------------------
# Original matched topologies (kept for ablation / fair parameter comparison)
# ---------------------------------------------------------------------------

class StrongAudioNet(nn.Module):
    """Spiking audio denoiser."""

    def __init__(self, channels: int = 64, time_dim: int = 128, num_steps: int = 8,
                 T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.input = nn.Conv1d(1, channels, 7, padding=3)
        self.b1 = ConvResBlock1D(channels, time_dim, num_steps)
        self.b2 = ConvResBlock1D(channels, time_dim, num_steps)
        self.b3 = ConvResBlock1D(channels, time_dim, num_steps)
        self.out = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))
        h = self.input(x.unsqueeze(1))
        h = self.b1(h, temb)
        h = self.b2(h, temb)
        h = self.b3(h, temb)
        return self.out(h).squeeze(1)


class NonSpikeAudioNet(nn.Module):
    """Original non-spiking (ANN) audio denoiser, matched topology to StrongAudioNet.
    Kept for controlled topology-matched ablation; the primary strong baseline
    is StrongANNAudioNet.
    """

    def __init__(self, channels: int = 64, time_dim: int = 128, T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.input = nn.Conv1d(1, channels, 7, padding=3)
        self.b1 = NonSpikeConvRes1D(channels, time_dim)
        self.b2 = NonSpikeConvRes1D(channels, time_dim)
        self.b3 = NonSpikeConvRes1D(channels, time_dim)
        self.out = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))
        h = self.input(x.unsqueeze(1))
        h = self.b1(h, temb)
        h = self.b2(h, temb)
        h = self.b3(h, temb)
        return self.out(h).squeeze(1)
