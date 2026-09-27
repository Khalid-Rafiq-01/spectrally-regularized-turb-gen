import torch
import torch.nn as nn
import torch.nn.functional as F


class Norm(nn.Module):
    """GroupNorm with the largest group count <= num_groups that divides num_channels."""
    def __init__(self, num_channels, num_groups=8):
        super().__init__()
        g = min(num_groups, num_channels)
        while num_channels % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, num_channels)

    def forward(self, x):
        return self.norm(x)


def circular_conv3x3(in_ch, out_ch, stride=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1,
                     padding_mode="circular", bias=bias)


def conv1x1(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)


class ResNetBlock(nn.Module):
    """Pre-activation residual block with circular padding and scaled skip."""
    def __init__(self, channels, hidden_dim=None, num_groups=4, res_scale=0.25):
        super().__init__()
        hidden_dim = hidden_dim or channels
        self.res_scale = res_scale
        self.conv_1 = nn.Sequential(
            Norm(channels, num_groups), nn.GELU(), circular_conv3x3(channels, hidden_dim),
            Norm(hidden_dim, num_groups), nn.GELU(), circular_conv3x3(hidden_dim, channels),
        )

    def forward(self, x):
        return x + self.res_scale * self.conv_1(x)


class Encoder_16(nn.Module):
    """1 x 256 x 256 -> z_channels x 16 x 16 (mean, logvar)."""
    def __init__(self, input_dim=1, z_channels=8):
        super().__init__()
        self.conv1 = circular_conv3x3(input_dim, 64, stride=1); self.rb1 = ResNetBlock(64, 64)
        self.conv2 = circular_conv3x3(64, 64, stride=2);        self.rb2 = ResNetBlock(64, 64)
        self.conv3 = circular_conv3x3(64, 128, stride=2);       self.rb3 = ResNetBlock(128, 128)
        self.down4 = circular_conv3x3(128, 128, stride=2);      self.rb4 = ResNetBlock(128, 128)
        self.down5 = circular_conv3x3(128, 256, stride=2);      self.rb5 = ResNetBlock(256, 256)
        self.to_mean   = conv1x1(256, z_channels)
        self.to_logvar = conv1x1(256, z_channels)

    def forward(self, x):
        x = self.rb1(self.conv1(x))   # 256
        x = self.rb2(self.conv2(x))   # 128
        x = self.rb3(self.conv3(x))   # 64
        x = self.rb4(self.down4(x))   # 32
        x = self.rb5(self.down5(x))   # 16
        mean   = self.to_mean(x)
        logvar = torch.clamp(self.to_logvar(x), -6.0, 2.0)
        return mean, logvar


class Decoder_16(nn.Module):
    """z_channels x 16 x 16 -> 1 x 256 x 256."""
    def __init__(self, z_channels=8, output_dim=1, num_groups=4):
        super().__init__()
        self.from_z = conv1x1(z_channels, 256)
        self.rb5 = ResNetBlock(256, 256, num_groups=num_groups); self.up5 = circular_conv3x3(256, 128)
        self.rb4 = ResNetBlock(128, 128, num_groups=num_groups); self.up4 = circular_conv3x3(128, 128)
        self.rb3 = ResNetBlock(128, 128, num_groups=num_groups); self.up3 = circular_conv3x3(128, 64)
        self.rb2 = ResNetBlock(64, 64, num_groups=num_groups);   self.up2 = circular_conv3x3(64, 64)
        self.rb1 = ResNetBlock(64, 64, num_groups=num_groups);   self.to_x = circular_conv3x3(64, output_dim)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, z):
        x = self.rb5(self.from_z(z))
        x = self.rb4(self._up(self.up5(x)))
        x = self.rb3(self._up(self.up4(x)))
        x = self.rb2(self._up(self.up3(x)))
        x = self.rb1(self._up(self.up2(x)))
        return self.to_x(x)


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        return mean + std * torch.randn_like(std)

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), z, mean, log_var


def build_vae(z_channels=8):
    """Architecture used for all three checkpoints in the paper."""
    return VAE(Encoder_16(1, z_channels), Decoder_16(z_channels, 1))
