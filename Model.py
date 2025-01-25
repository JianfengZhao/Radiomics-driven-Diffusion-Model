import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import MambaBlock, TransformerModule, UpSampleFunction, DownSampleFunction

class DiffusionModel(nn.Module):
    def __init__(self, num_timesteps, model, beta_start=1e-4, beta_end=2e-2):
        super(DiffusionModel, self).__init__()
        self.num_timesteps = num_timesteps
        self.model = model

        # Linearly spaced betas for the forward process
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x_0):
        """
        Perform the full forward diffusion process (adding noise).
        This simulates the diffusion from x_0 to x_T with intermediate steps.
        """
        batch_size = x_0.size(0)
        x_t = x_0
        noise_sequence = []

        for t in range(self.num_timesteps):
            noise = torch.randn_like(x_0)
            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1, 1)  # Adjusted for 3D
            x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
            noise_sequence.append(x_t)  # Save the noisy image at each step

        return x_t, noise_sequence

    def reverse(self, x_T, X_C, R, mamba_blocks, transformer, upsample, downsample):
        """
        Perform the full reverse denoising process using the defined modules.
        This reverses the noise process, starting from x_T and progressing to x_0.
        """
        x_t = x_T
        for t in reversed(range(self.num_timesteps)):
            x_t = self.denoise_step(x_t, X_C, R, t, mamba_blocks, transformer, upsample, downsample)

        return x_t

    def denoise_step(self, x_t, X_C, R, t, mamba_blocks, transformer, upsample, downsample):
        """
        Perform a single denoising step at time t using the modules.
        """
        # Concatenate the conditional image with the current noisy image
        input_concat = torch.cat([x_t, X_C], dim=1)  # Expecting 3D volumes

        # Split into 3D patches and input to the UNet's Mamba Block
        input_patches = self.split_to_patches_3d(input_concat)
        x = input_patches

        # Encoder part: 4 MambaBlocks, each includes downsampling
        for block in mamba_blocks[:4]:
            x = block(x)
            x = downsample(x)  # From 1D to 3D, perform downsampling, then from 3D to 1D

        # Cross-attention integration with Transformer
        phi_R = transformer(x, R)  # Use transformer to combine image and radiomics features
        x = transformer.cross_attention(x, phi_R, phi_R)

        # Decoder part: 4 MambaBlocks, each includes upsampling
        for block in mamba_blocks[4:]:
            x = upsample(x)  # Reverse of the downsampling process
            x = block(x)

        # Reconstruct the denoised image
        x_t_minus_1 = self.reconstruct_from_patches_3d(x)
        return x_t_minus_1

    def sample(self, x_T, X_C, R, mamba_blocks, transformer, upsample, downsample):
        """
        Generate samples from the model using the reverse process.
        This simulates the generation of images from pure noise.
        """
        return self.reverse(x_T, X_C, R, mamba_blocks, transformer, upsample, downsample)

    def mccs_sampling(self, x_0, j, I, C, X_C, R, mamba_blocks, transformer, upsample, downsample):
        """
        Monte Carlo Compression Sampling (MCCS) integrated within DiffusionModel.
        """
        predictions = []

        for iteration in range(I):
            x_t = x_0.clone()
            current_step = self.num_timesteps

            intervals = [int(np.random.uniform(self.num_timesteps / (j - 1) - C, self.num_timesteps / (j - 1) + C)) for _ in range(j - 1)]
            intervals = np.clip(intervals, 1, None)
            intervals.insert(0, 0)
            intervals.append(self.num_timesteps - sum(intervals))

            sampling_points = np.cumsum(intervals)

            for step in reversed(sampling_points):
                while current_step > step:
                    x_t = self.model(x_t, current_step)
                    current_step -= 1

                x_t = self.denoise_step(x_t, X_C, R, step, mamba_blocks, transformer, upsample, downsample)
                current_step = step - 1

            predictions.append(x_t.cpu().detach().numpy())

        predictions = np.array(predictions)
        x_hat_0 = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)

        return x_hat_0, variance

    def split_to_patches_3d(self, x):
        batch_size, channels, depth, height, width = x.size()
        patch_size = 16
        x = x.view(batch_size, channels, 
                   depth // patch_size, patch_size, 
                   height // patch_size, patch_size, 
                   width // patch_size, patch_size)
        x = x.permute(0, 2, 4, 6, 1, 3, 5).contiguous()
        return x.view(batch_size, -1, patch_size, patch_size, patch_size)

    def reconstruct_from_patches_3d(self, x):
        batch_size, num_patches, patch_size, _, _ = x.size()
        depth = height = width = int(num_patches ** (1/3)) * patch_size
        x = x.view(batch_size, 
                   depth // patch_size, height // patch_size, width // patch_size,
                   -1, patch_size, patch_size, patch_size)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.view(batch_size, -1, depth, height, width)