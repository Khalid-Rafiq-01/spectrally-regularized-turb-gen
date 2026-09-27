"""Encode the training set with the frozen VAE encoder mean (Stage 2 input).

Run from the repository root:
    python -m generation.encode_latents --vae_name A_lat16_mse
    python -m generation.encode_latents --vae_name B_lat16_weighted_mse_spec
"""
import os
import argparse
import numpy as np
import torch

from generation.sampling import load_vae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_name", required=True)
    p.add_argument("--vae_dir", default="./checkpoints/compression")
    p.add_argument("--data_path", default="./data/trajectory_real.npy")
    p.add_argument("--out_dir", default="./data/latents")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, mu, sigma = load_vae(f"{args.vae_dir}/best_{args.vae_name}.pt", device)

    train_raw = np.load(args.data_path)[1000:5500]
    train_norm = (train_raw - mu) / (sigma + 1e-9)

    means = []
    with torch.no_grad():
        for i in range(0, len(train_norm), args.batch_size):
            xb = torch.tensor(train_norm[i:i + args.batch_size, None], dtype=torch.float32, device=device)
            means.append(vae.encoder(xb)[0].cpu().numpy())
    z = np.concatenate(means)

    os.makedirs(args.out_dir, exist_ok=True)
    out = f"{args.out_dir}/train_latents_{args.vae_name}.npy"
    np.save(out, z)
    print(f"{args.vae_name}: {z.shape}  std={z.std():.4f}  ->  {out}")


if __name__ == "__main__":
    main()
