"""Train the Stage 2 latent flow matching model (CondOT path) on frozen VAE latents.

Run from the repository root:
    python -m generation.train_fm --vae_name A_lat16_mse
    python -m generation.train_fm --vae_name B_lat16_weighted_mse_spec
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from generation.unet import Unet
from generation.sampling import Z_CHANNELS, EMBED_DIM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_name", required=True, help="selects data/latents/train_latents_<vae_name>.npy")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent_dir", default="./data/latents")
    p.add_argument("--vae_dir", default="./checkpoints/compression")
    p.add_argument("--save_dir", default="./checkpoints/generation")
    p.add_argument("--no_circular_pad", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    circular_pad = not args.no_circular_pad
    name = args.vae_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    z_np = np.load(f"{args.latent_dir}/train_latents_{name}.npy")
    sigma = float(torch.load(f"{args.vae_dir}/best_{name}.pt", map_location="cpu")["sigma"])
    print(f"Training FM — {name}  (latents {z_np.shape}, std={z_np.std():.4f}, circular_pad={circular_pad})")

    loader = DataLoader(TensorDataset(torch.tensor(z_np, dtype=torch.float32)),
                        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    unet = Unet(input_dim=Z_CHANNELS, output_dim=Z_CHANNELS, embedding_dim=EMBED_DIM,
                circular_pad=circular_pad).to(device)
    print(f"  UNet params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")

    opt = Adam(unet.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler("cuda")
    loss_fn = nn.MSELoss()

    best_loss, history = float("inf"), []
    ckpt_best = f"{args.save_dir}/best_fm_{name}.pt"
    meta = {"name": name, "sigma": sigma, "circular_pad": circular_pad}

    for epoch in range(args.epochs):
        unet.train()
        epoch_loss = 0.0
        for (z1,) in loader:
            z1 = z1.to(device)
            eps = torch.randn_like(z1)
            t_scalar = torch.rand(z1.shape[0], device=device)
            t = t_scalar[:, None, None, None]
            z_t = t * z1 + (1.0 - t) * eps        # CondOT interpolant
            target = z1 - eps                     # target velocity

            opt.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = loss_fn(unet(z_t, t_scalar), target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        sched.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state": unet.state_dict(), "epoch": epoch + 1, "loss": best_loss, **meta}, ckpt_best)

        if (epoch + 1) % 500 == 0:
            ckpt_ep = f"{args.save_dir}/fm_{name}_ep{epoch+1}.pt"
            torch.save({"model_state": unet.state_dict(), "epoch": epoch + 1, "loss": avg_loss, **meta}, ckpt_ep)
            print(f"  Checkpoint saved -> {ckpt_ep}")

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Ep {epoch+1:04d}/{args.epochs}  |  loss {avg_loss:.5f}  |  best {best_loss:.5f}")

    np.save(f"{args.save_dir}/fm_history_{name}.npy", np.array(history))
    print(f"  Best loss = {best_loss:.6f}  ->  {ckpt_best}")


if __name__ == "__main__":
    main()
