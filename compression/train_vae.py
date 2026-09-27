"""Train the Stage 1 VAE.

Run from the repository root:
    python -m vae.train_vae --name A_lat16_mse
    python -m vae.train_vae --name B_lat16_mse_spec          --w_ir 4e-3 --w_do 4e-3 --w_dd 4e-3
    python -m vae.train_vae --name B_lat16_weighted_mse_spec --w_ir 1e-3 --w_do 4e-3 --w_dd 6e-3
"""
import os
import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from compression.models import build_vae
from compression.losses import mse_kld_loss, spectral_loss_zones


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=None, help="checkpoint saved as <save_dir>/best_<name>.pt")
    p.add_argument("--w_ir", type=float, default=0.0)
    p.add_argument("--w_do", type=float, default=0.0)
    p.add_argument("--w_dd", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=7.5e-4)
    p.add_argument("--beta_kld", type=float, default=7.5e-3)
    p.add_argument("--data_path", default="./data/trajectory_real.npy")
    p.add_argument("--save_dir", default="./checkpoints")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (required for sweeps)")
    p.add_argument("--project", default="LatentFM_VAE")
    return p.parse_args()


def load_data(path):
    data = np.load(path)
    train_raw, test_raw = data[1000:5500], data[5500:6000]   # drop spin-up; temporal split
    mu, sigma = train_raw.mean(), train_raw.std()
    return (train_raw - mu) / (sigma + 1e-9), (test_raw - mu) / (sigma + 1e-9), mu, sigma


def latent_stats(model, loader, device, n_max=500):
    """Per-channel std of encoder means; channels < 0.05 are flagged as collapsed."""
    model.eval()
    all_mu, n = [], 0
    with torch.no_grad():
        for (xb,) in loader:
            mu_z, _ = model.encoder(xb.to(device))
            all_mu.append(mu_z.float().cpu())
            n += xb.size(0)
            if n >= n_max:
                break
    return torch.cat(all_mu, 0).std(dim=(0, 2, 3))


def run_epoch(model, loader, device, w, beta, opt=None, scaler=None):
    """One pass over loader. Trains if opt is given, otherwise evaluates with the encoder mean."""
    train = opt is not None
    model.train(train)
    keys = ["rl", "kld", "ir", "do", "dd", "total"]
    sums, n = dict.fromkeys(keys, 0.0), 0

    with torch.set_grad_enabled(train):
        for (xb,) in loader:
            xb = xb.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    x_hat, _, mean, lv = model(xb)
            else:
                with autocast("cuda"):
                    mean, lv = model.encoder(xb)
                    x_hat = model.decoder(mean)

            x_hat = x_hat.float()
            rl, kld = mse_kld_loss(xb, x_hat, mean.float(), lv.float())
            L_ir, L_do, L_dd = spectral_loss_zones(xb, x_hat)
            total = rl + beta * kld + w[0] * L_ir + w[1] * L_do + w[2] * L_dd

            if train:
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

            bs = xb.size(0)
            for k, v in zip(keys, [rl, kld, L_ir, L_do, L_dd, total]):
                sums[k] += v.item() * bs
            n += bs

    return {k: v / n for k, v in sums.items()}


def main():
    args = parse_args()
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.project, config=vars(args))
        for k in ("w_ir", "w_do", "w_dd"):          # sweep values override CLI defaults
            setattr(args, k, run.config.get(k, getattr(args, k)))

    w = (args.w_ir, args.w_do, args.w_dd)
    name = args.name or f"wir{w[0]:.0e}_wdo{w[1]:.0e}_wdd{w[2]:.0e}"
    if run is not None:
        run.name = name
    print(f"\nRun: {name}   weights (IR, DO, DD) = {w}")

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_norm, test_norm, mu, sigma = load_data(args.data_path)
    make = lambda a, shuffle: DataLoader(TensorDataset(torch.tensor(a[:, None], dtype=torch.float32)),
                                         batch_size=args.batch_size, shuffle=shuffle,
                                         num_workers=4, pin_memory=True)
    train_loader, test_loader = make(train_norm, True), make(test_norm, False)

    model  = build_vae().to(device)
    opt    = Adam(model.parameters(), lr=args.lr)
    sched  = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler("cuda")

    best_val, collapsed_flag = float("inf"), False
    best_ckpt = os.path.join(args.save_dir, f"best_{name}.pt")

    for epoch in range(args.epochs):
        tr = run_epoch(model, train_loader, device, w, args.beta_kld, opt, scaler)
        sched.step()
        va = run_epoch(model, test_loader, device, w, args.beta_kld)

        log = {"epoch": epoch + 1,
               **{f"train_{k}": v for k, v in tr.items()},
               "val_rl": va["rl"], "val_kld": va["kld"], "val_spec_ir": va["ir"],
               "val_spec_do": va["do"], "val_spec_dd": va["dd"], "val_total": va["total"]}

        if (epoch + 1) % 50 == 0:
            ch_std = latent_stats(model, test_loader, device)
            n_collapsed = (ch_std < 0.05).sum().item()
            collapsed_flag |= n_collapsed > 0
            log.update(min_latent_std=ch_std.min().item(), n_collapsed=n_collapsed,
                       collapsed=int(collapsed_flag))
            print(f"  Ep {epoch+1:03d}  val_total={va['total']:.5f}  val_dd={va['dd']:.5f}"
                  f"  min_lat_std={ch_std.min().item():.3f}  collapsed={n_collapsed}")
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Ep {epoch+1:03d}  val_total={va['total']:.5f}  val_ir={va['ir']:.5f}"
                  f"  val_do={va['do']:.5f}  val_dd={va['dd']:.5f}")

        if run is not None:
            run.log(log)

        if va["total"] < best_val:
            best_val = va["total"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "val_total": best_val,
                        "mu": mu, "sigma": sigma, "w_ir": w[0], "w_do": w[1], "w_dd": w[2]}, best_ckpt)

    print(f"\nBest val_total = {best_val:.5f}  ->  {best_ckpt}")
    if run is not None:
        run.summary.update(best_val_total=best_val, collapsed=int(collapsed_flag), best_ckpt=best_ckpt)
        run.finish()


if __name__ == "__main__":
    main()
