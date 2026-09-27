import numpy as np


def bandpass(field, k_lo, k_hi):
    """Keep Fourier modes with k_lo <= |k| < k_hi."""
    H, W = field.shape
    fhat = np.fft.rfft2(field)
    ky = np.fft.fftfreq(H) * H
    kx = np.arange(W // 2 + 1)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(KX**2 + KY**2)
    fhat[(K < k_lo) | (K >= k_hi)] = 0.0
    return np.fft.irfft2(fhat, s=(H, W)).real


def isotropic_spectrum(field_np):
    """Shell-averaged power spectrum. Returns (E_k, k) for k = 1 .. min(H, W)//2."""
    H, W = field_np.shape
    k_max = min(H, W) // 2
    power = np.abs(np.fft.rfft2(field_np)) ** 2
    ky = np.fft.fftfreq(H) * H
    kx = np.arange(W // 2 + 1)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k_int = np.round(np.sqrt(KX**2 + KY**2)).astype(int)
    rfft_fac = np.ones_like(power)
    rfft_fac[:, 1:-1] = 2.0
    E_k = np.zeros(k_max)
    for k in range(1, k_max + 1):
        mask = k_int == k
        if mask.any():
            E_k[k - 1] = (rfft_fac[mask] * power[mask]).sum() / mask.sum()
    return E_k, np.arange(1, k_max + 1)
