# Data

`trajectory_real.npy` (~1.6 GB, shape `(6000, 256, 256)`, float) is hosted at:

**[DATASET_LINK]**

Download it into this folder:

```
data/
├── trajectory_real.npy
└── latents/            # created by generation/encode_latents.py
```

The field is vorticity from 2D forced Navier–Stokes (jax-cfd), $\nu = 10^{-3}$, $k_f = 4$, $256^2$ grid, $\Delta t = 5\times10^{-4}$, one snapshot every 200 solver steps. Snapshots 0–999 are spin-up, 1000–5499 train, 5500–5999 test.
