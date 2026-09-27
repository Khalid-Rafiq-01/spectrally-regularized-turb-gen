import torch
import torch.nn.functional as F


def mse_kld_loss(x, x_hat, mean, log_var):
    recon = F.mse_loss(x_hat, x)
    kld = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    kld = kld.mean(dim=(1, 2, 3)).mean()
    return recon, kld


_SHELL_CACHE = {}


def _shell_masks(H, W, k_max, device):
    """Integer-wavenumber shell masks on the rfft2 grid, built once per (H, W, k_max, device)."""
    key = (H, W, k_max, str(device))
    if key not in _SHELL_CACHE:
        ky = torch.fft.fftfreq(H) * H
        kx = torch.fft.rfftfreq(W) * W
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K = torch.sqrt(KX.square() + KY.square()).to(device)
        masks = torch.stack([((K >= k - 0.5) & (K < k + 0.5)).float() for k in range(1, k_max + 1)])
        counts = masks.sum(dim=(1, 2)).clamp(min=1.0)
        _SHELL_CACHE[key] = (masks, counts)   # (k_max, H, W//2+1), (k_max,)
    return _SHELL_CACHE[key]


def spectral_loss_zones(x, x_hat, k_ir=(6, 40), k_do=(40, 65), k_dd=(65, 85),
                        k_max=85, eps=1e-6, clamp_max=50.0):
    """Zone-wise mean squared log-spectral error (Eq. 4). Returns (L_ir, L_do, L_dd).

    Shell k maps to index k-1, so the zones cover k=6-40, 41-65 and 66-85.
    """
    def power_rfft(field):
        f = torch.fft.rfft2(field[:, 0])
        p = f.real.square() + f.imag.square()
        if p.shape[-1] > 2:
            p[..., 1:-1] = 2.0 * p[..., 1:-1]
        return p

    H, W = x.shape[-2:]
    masks, counts = _shell_masks(H, W, k_max, x.device)

    Ex   = torch.einsum("khw,bhw->bk", masks, power_rfft(x)) / counts
    Ehat = torch.einsum("khw,bhw->bk", masks, power_rfft(x_hat)) / counts
    sq_err = (torch.log(Ehat + eps) - torch.log(Ex + eps)).pow(2)   # (B, k_max)

    L_ir = torch.clamp(sq_err[:, k_ir[0] - 1:k_ir[1]].mean(), max=clamp_max)
    L_do = torch.clamp(sq_err[:, k_do[0]:k_do[1]].mean(), max=clamp_max)
    L_dd = torch.clamp(sq_err[:, k_dd[0]:k_dd[1]].mean(), max=clamp_max)
    return L_ir, L_do, L_dd
