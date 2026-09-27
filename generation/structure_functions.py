import numpy as np


def longitudinal_structure_functions(vorticity_hat, nx, ny, r_vals=None, n_angles=16, orders=(2, 3)):
    """S_p(r) = <[delta u_L(r)]^p> from rfft2 vorticity. Velocity via psi_hat = -omega_hat / k^2."""
    if r_vals is None:
        dense = np.arange(1, min(21, nx // 2))
        coarse = np.unique(np.round(np.logspace(np.log10(20), np.log10(nx // 2), 30)).astype(int))
        r_vals = np.unique(np.concatenate([dense, coarse[coarse > 20]]))
        r_vals = r_vals[r_vals <= nx // 2]

    kx = np.fft.fftfreq(nx, d=1.0 / nx).astype(float)
    ky = np.arange(ny // 2 + 1, dtype=float)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    psi_hat = -vorticity_hat / K2
    psi_hat[0, 0] = 0.0
    ux = np.fft.irfft2(1j * KY * psi_hat, s=(nx, ny))
    uy = np.fft.irfft2(-1j * KX * psi_hat, s=(nx, ny))

    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    SF = {p: np.zeros(len(r_vals)) for p in orders}
    for ri, r in enumerate(r_vals):
        accum = {p: [] for p in orders}
        for theta in angles:
            dx = int(round(r * np.cos(theta))) % nx
            dy = int(round(r * np.sin(theta))) % ny
            ex, ey = np.cos(theta), np.sin(theta)
            delta_uL = ((np.roll(np.roll(ux, -dx, axis=0), -dy, axis=1) - ux) * ex
                        + (np.roll(np.roll(uy, -dx, axis=0), -dy, axis=1) - uy) * ey)
            for p in orders:
                accum[p].append(np.mean(delta_uL**p))
        for p in orders:
            SF[p][ri] = np.mean(accum[p])

    r_phys = np.array(r_vals, dtype=float) * (2.0 * np.pi / nx)
    return r_phys, SF


def compute_sf_ensemble(fields_phys, n_fields=500, n_angles=16, orders=(2, 3)):
    """Per-field S_p(r) for the first n_fields of an (N, nx, ny) vorticity ensemble."""
    nx, ny = fields_phys.shape[1:]
    n_fields = min(n_fields, len(fields_phys))
    SF_all = None
    for i in range(n_fields):
        r_phys, SF_i = longitudinal_structure_functions(np.fft.rfft2(fields_phys[i]), nx, ny,
                                                        n_angles=n_angles, orders=orders)
        if SF_all is None:
            SF_all = {p: np.zeros((n_fields, len(r_phys))) for p in orders}
        for p in orders:
            SF_all[p][i] = SF_i[p]
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_fields} done")
    return r_phys, SF_all
