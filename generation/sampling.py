import time
import numpy as np
import torch

from compression.models import Encoder_16, Decoder_16, VAE
from generation.unet import Unet

Z_CHANNELS = 8
EMBED_DIM = 64


# Loading
def load_vae(path, device):
    """Frozen Stage 1 VAE. Returns (vae, mu, sigma) with the data normalization stored in the checkpoint."""
    vae = VAE(Encoder_16(1, Z_CHANNELS), Decoder_16(Z_CHANNELS, 1)).to(device)
    ck = torch.load(path, map_location=device)
    vae.load_state_dict(ck["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, float(ck["mu"]), float(ck["sigma"])


def load_fm(path, device, circular_pad=True):
    """Frozen Stage 2 velocity field. circular_pad must match training."""
    ck = torch.load(path, map_location=device)
    unet = Unet(input_dim=Z_CHANNELS, output_dim=Z_CHANNELS, embedding_dim=EMBED_DIM,
                circular_pad=circular_pad).to(device)
    unet.load_state_dict(ck["model_state"])
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    return unet


# Fixed-step ODE solvers, integrating dz/dtau = v(z, tau) from tau=0 to 1
@torch.no_grad()
def euler(unet, z0, n_steps=100):
    """NFE = n_steps."""
    z, dt = z0.clone(), 1.0 / n_steps
    for i in range(n_steps):
        z = z + unet(z, torch.full((z.shape[0],), i * dt, device=z.device)) * dt
    return z


@torch.no_grad()
def heun(unet, z0, n_steps=100):
    """NFE = 2 * n_steps."""
    z, dt = z0.clone(), 1.0 / n_steps
    for i in range(n_steps):
        tv0 = torch.full((z.shape[0],), i * dt, device=z.device)
        tv1 = torch.full((z.shape[0],), (i + 1) * dt, device=z.device)
        k1 = unet(z, tv0)
        k2 = unet(z + dt * k1, tv1)
        z = z + dt * 0.5 * (k1 + k2)
    return z


@torch.no_grad()
def rk4(unet, z0, n_steps=100):
    """NFE = 4 * n_steps."""
    z, dt = z0.clone(), 1.0 / n_steps
    for i in range(n_steps):
        tv0 = torch.full((z.shape[0],), i * dt, device=z.device)
        tvm = torch.full((z.shape[0],), (i + 0.5) * dt, device=z.device)
        tv1 = torch.full((z.shape[0],), (i + 1.0) * dt, device=z.device)
        k1 = unet(z, tv0)
        k2 = unet(z + 0.5 * dt * k1, tvm)
        k3 = unet(z + 0.5 * dt * k2, tvm)
        k4 = unet(z + dt * k3, tv1)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


# Sampling helpers
@torch.no_grad()
def decode_batched(decoder, z, sigma, mu, bs=10):
    """Decode latents to physical vorticity in small batches."""
    return np.concatenate([decoder(z[i:i + bs])[:, 0].cpu().numpy() * sigma + mu
                           for i in range(0, len(z), bs)], axis=0)


@torch.no_grad()
def calibrate_scale(unet, lat_std, device, n_cal=500, n_steps=100):
    """Prior scale T such that Euler samples match the training-latent std (Sec. 4.3)."""
    z0 = torch.randn(n_cal, Z_CHANNELS, 16, 16, device=device)
    return lat_std / euler(unet, z0, n_steps).cpu().numpy().std()


@torch.no_grad()
def time_solver(solver_fn, unet, z0, n_steps, n_warmup=3, n_repeat=10):
    """Wall-clock (ms) mean and std of one solver call."""
    for _ in range(n_warmup):
        solver_fn(unet, z0.clone(), n_steps)
    times = []
    if z0.device.type == "cuda":
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(n_repeat):
            start.record()
            solver_fn(unet, z0.clone(), n_steps)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            solver_fn(unet, z0.clone(), n_steps)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))
