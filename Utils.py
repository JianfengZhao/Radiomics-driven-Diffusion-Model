import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_state=16, d_conv=4, expand=2, dt_rank="auto", 
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, 
                 conv_bias=True, bias=False, nslices=5, device=None, dtype=None):
        super(MambaBlock, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.input_dim)
        self.dt_rank = math.ceil(self.input_dim / 16) if dt_rank == "auto" else dt_rank
        self.nslices = nslices

        # Projections
        self.in_proj = nn.Linear(self.input_dim, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Convolutional layers
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                kernel_size=self.d_conv, groups=self.d_inner,
                                padding=self.d_conv - 1, bias=conv_bias, **factory_kwargs)
        self.conv1d_b = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                  kernel_size=self.d_conv, groups=self.d_inner,
                                  padding=self.d_conv - 1, bias=conv_bias, **factory_kwargs)
        self.conv1d_s = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                  kernel_size=self.d_conv, groups=self.d_inner,
                                  padding=self.d_conv - 1, bias=conv_bias, **factory_kwargs)

        # Activation function
        self.act = nn.SiLU()

        # State space model parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.x_proj_s = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj_s = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj_b.bias.copy_(inv_dt)
            self.dt_proj_s.bias.copy_(inv_dt)

        # SSM Initialization
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        A_b = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                     "n -> d n", d=self.d_inner).contiguous()
        self.A_b_log = nn.Parameter(torch.log(A_b))
        self.A_b_log._no_weight_decay = True
        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_b._no_weight_decay = True

        A_s = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                     "n -> d n", d=self.d_inner).contiguous()
        self.A_s_log = nn.Parameter(torch.log(A_s))
        self.A_s_log._no_weight_decay = True
        self.D_s = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_s._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.input_dim, bias=bias, **factory_kwargs)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initial projection
        xz = rearrange(self.in_proj(x), "b l d -> b d l", l=seq_len)

        # Split projections
        x, z = xz.chunk(2, dim=1)

        # Forward pass through convolutional layers
        x = self.act(self.conv1d(x)[..., :seq_len])
        x_b = self.act(self.conv1d_b(x.flip([-1]))[..., :seq_len])
        x_s = self.conv1d_s(x)

        # SSM step
        A = -torch.exp(self.A_log.float())
        A_b = -torch.exp(self.A_b_log.float())
        A_s = -torch.exp(self.A_s_log.float())

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        x_db = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))
        x_ds = self.x_proj_s(rearrange(x_s, "b d l -> (b l) d"))

        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_b, B_b, C_b = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_s, B_s, C_s = torch.split(x_ds, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Final output projection
        out = self.out_proj(x + x_b.flip([-1]) + x_s)

        # Reshape output back to 3D if needed (this depends on how the input was originally reshaped)
        # Assuming original input was [batch_size, channels, depth, height, width]
        out = rearrange(out, "b d l -> b l d")

        return out


class TransformerModule(nn.Module):
    def __init__(self, img_feature_dim, radiomics_feature_dim, num_heads, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerModule, self).__init__()
        
        # Linear projection for the image and radiomics features
        self.W_Q = nn.Linear(img_feature_dim, img_feature_dim)
        self.W_K = nn.Linear(radiomics_feature_dim, img_feature_dim)
        self.W_V = nn.Linear(radiomics_feature_dim, img_feature_dim)

        # Positional encoding for the image features
        self.pos_encoder = PositionalEncoding(img_feature_dim, dropout)
        
        # Transformer encoder layer
        encoder_layers = TransformerEncoderLayer(d_model=img_feature_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Final linear projection back to the img_feature_dim
        self.out_proj = nn.Linear(img_feature_dim, img_feature_dim)

    def forward(self, img_features, radiomics_features):
        """
        img_features: Image features, used as Query (Q)
        radiomics_features: Radiomics features, used as Key (K) and Value (V)
        """
        # Linear projections
        Q = self.W_Q(img_features)
        K = self.W_K(radiomics_features)
        V = self.W_V(radiomics_features)
        
        # Expand dimensions to match Transformer input requirements
        K = K.unsqueeze(1).repeat(1, Q.size(1), 1)  # Expand radiomics features to match image features
        V = V.unsqueeze(1).repeat(1, Q.size(1), 1)

        # Add positional encoding to image features
        Q = self.pos_encoder(Q)

        # Compute attention mechanism
        attention_output = self.transformer_encoder(Q + K)  # Using K as context here, can be adjusted as needed
        
        # Output projection
        output = self.out_proj(attention_output + V)  # Adding V to introduce some radiomics-based variance
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class UpSampleFunction(nn.Module):
    def __init__(self, scale_factor=2, mode='trilinear'):
        super(UpSampleFunction, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """
        x: Input tensor with shape [batch_size, channels, depth, height, width]
        """
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)


class DownSampleFunction(nn.Module):
    def __init__(self, scale_factor=0.5, mode='trilinear'):
        super(DownSampleFunction, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """
        x: Input tensor with shape [batch_size, channels, depth, height, width]
        """
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)