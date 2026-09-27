import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterleavedPositionalEncoding(nn.Module):
    """Sinusoidal embedding of the flow time tau in [0, 1]."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, circular_pad=False):
        super().__init__()
        pad_mode = "circular" if circular_pad else "zeros"
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode=pad_mode)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode=pad_mode)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(torch.relu(self.norm1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.conv2(torch.relu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels_in, channels_proj):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels_in)
        self.q = nn.Conv2d(channels_in, channels_proj, 1)
        self.k = nn.Conv2d(channels_in, channels_proj, 1)
        self.v = nn.Conv2d(channels_in, channels_in, 1)

    def forward(self, x):
        h = self.norm(x)
        Q, K, V = self.q(h), self.k(h), self.v(h)
        B, P, H, W = Q.shape
        C = x.shape[1]
        N = H * W
        Q = Q.view(B, P, N).permute(0, 2, 1)
        K = K.view(B, P, N).permute(0, 2, 1)
        V = V.view(B, C, N).permute(0, 2, 1)
        attn = F.softmax(Q @ K.transpose(-2, -1) / (P ** 0.5), dim=-1)
        return (attn @ V).permute(0, 2, 1).view(B, C, H, W)


class Unet(nn.Module):
    """Velocity field v_theta(z, tau) for latent flow matching. Input/output: (B, Z_CHANNELS, 16, 16).

    circular_pad=True uses circular padding in all 3x3 convolutions, matching the periodic VAE latents.

    Skips: h2 (B, 64, 16, 16) after Down1, h3 (B, 128, 16, 16) after Down2.
    Up1: upsample(h6) cat h3 -> 256 -> 128.  Up2: h8 cat h2 -> 192 -> 64.
    """
    def __init__(self, input_dim, embedding_dim, output_dim, circular_pad=False):
        super().__init__()
        self.input_dim, self.embedding_dim, self.output_dim = input_dim, embedding_dim, output_dim
        self.circular_pad = circular_pad
        pad_mode = "circular" if circular_pad else "zeros"

        self.conv1 = nn.Conv2d(input_dim, 64, 3, padding=1, padding_mode=pad_mode)
        self.time_embedding = nn.Sequential(
            InterleavedPositionalEncoding(embedding_dim),
            nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Linear(256, 256))

        self.ResBlock_Down1 = ResidualBlock(64, 64, 256, circular_pad)
        self.ResBlock_Down2 = ResidualBlock(64, 128, 256, circular_pad)
        self.AvgPool = nn.AvgPool2d(2, 2)

        self.ResBlock_Mid1 = ResidualBlock(128, 128, 256, circular_pad)
        self.attention_block = SelfAttention(128, 128)
        self.ResBlock_Mid2 = ResidualBlock(128, 128, 256, circular_pad)

        self.ResBlock_Up1 = ResidualBlock(256, 128, 256, circular_pad)
        self.ResBlock_Up2 = ResidualBlock(192, 64, 256, circular_pad)
        self.final_conv = nn.Conv2d(64, output_dim, 3, padding=1, padding_mode=pad_mode)

    def forward(self, x, t):
        te = self.time_embedding(t)
        h1 = self.conv1(x)
        h2 = self.ResBlock_Down1(h1, te)
        h3 = self.ResBlock_Down2(h2, te)
        h4 = self.AvgPool(h3)

        h5 = self.ResBlock_Mid1(h4, te)
        h5 = h5 + self.attention_block(h5)
        h6 = self.ResBlock_Mid2(h5, te)

        h7 = torch.cat((F.interpolate(h6, size=(16, 16), mode="nearest"), h3), dim=1)
        h8 = self.ResBlock_Up1(h7, te)
        h10 = self.ResBlock_Up2(torch.cat((h8, h2), dim=1), te)
        return self.final_conv(h10)
