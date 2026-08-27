# Spiking vs Non-Spiking Residual Denoising

Controlled comparison of **spiking neural networks (SNNs)** and **matched / strong ANN baselines** for residual denoising on:

- **Images** — MNIST + standard DDPM Gaussian diffusion  
- **Audio** — Google SpeechCommands + **Markov Poisson diffusion**

## Key upgrades (vs. original codebase)

1. **Proper Markov Poisson diffusion (audio)**  
   Forward process is now a true diffusion Markov chain  
   \(q(x_t \mid x_{t-1})\) with closed-form \(q(x_t \mid x_0)\) via cumulative thinning (`gamma_bar`).  
   This replaces the previous non-Markov one-shot corruption \(x_t = f(x_0, t)\).

2. **Rigorous energy analysis**  
   Per-layer MAC / AC counts converted to energy with literature 45 nm CMOS constants  
   (Horowitz ISSCC 2014: \(E_\text{MAC} \approx 45\,\text{pJ}\), \(E_\text{AC} \approx 0.9\,\text{pJ}\)).  
   Reports absolute µJ and ANN/SNN ratio.

3. **Forward-process ablation**  
   Same SNN architecture trained under  
   - Poisson diffusion  
   - Gaussian additive noise  
   - Bernoulli spike encoding  
   Justifies (or falsifies) the Poisson choice with empirical evidence.

4. **Temporal-conditioning visualization**  
   Denoising behaviour plotted at \(t \in \{0,5,10,20,30,39\}\).  
   Qualitative evidence that the network actually uses the timestep embedding.

5. **Stronger ANN baseline**  
   Primary non-spiking competitor is a **dilated residual TCN** (6 blocks, dilations 1…32, 96 channels, multi-scale fusion).  
   Topology-matched residual ANN is still trained for completeness.  
   If the SNN remains competitive, the result carries more weight.

## Quick start

```bash
pip install -r requirements.txt
python main.py                     # full experiment
python main.py --skip-audio        # image only
python main.py --skip-image        # audio only
python main.py --epochs-img 3 --epochs-audio 3 --ablation-epochs 2 --skip-ablation  # smoke test
```

Figures land in `outputs/figures/`:

- `image_samples.png`
- `audio_waveforms.png`
- `temporal_conditioning.png`
- `corruption_ablation.png`
- `comparison_bars.png`

## Project layout

```
main.py
requirements.txt
src/
  diffusion_image.py   # DDPM + sinusoidal time embedding
  diffusion_audio.py   # PoissonDiffusion + Gaussian/Bernoulli corruptions
  models_image.py
  models_audio.py      # StrongAudioNet (SNN), StrongANNAudioNet (dilated TCN), matched ANN
  datasets.py
  train.py
  evaluate.py          # paired eval, sparsity, timing, energy
  metrics.py           # SI-SDR, SNR, energy helpers
  visualize.py
```

## Citation notes for energy numbers

- M. Horowitz, “1.1 Computing’s energy problem (and what we can do about it),” *ISSCC*, 2014.  
- Typical neuromorphic references: Merolla et al. (TrueNorth), Davies et al. (Loihi).  
  Real spike-based chips can be 10–100× more efficient than the conservative digital model used here; substitute chip-specific constants as needed.
