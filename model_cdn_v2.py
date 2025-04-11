# 文件名: model_cdn_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================================================
# 0. 基础构建模块 (来自你的原始代码, PatchEmbedding/PositionalEncoding 稍作调整)
# ==================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50): # 减少 max_len for shorter sequences
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Adjust PE size and transpose correctly for addition
        pe_to_add = self.pe[:x.size(1), :] # Shape: (seq_len, 1, d_model)
        pe_to_add = pe_to_add.permute(1, 0, 2) # Shape: (1, seq_len, d_model)
        x = x + pe_to_add # Add using broadcasting
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, patch_size=8, stride=4, ppg_length=64):
        super().__init__()
        padding = (patch_size - stride) // 2 # Try to maintain length somewhat
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.num_patches = self._get_output_length(ppg_length, padding, patch_size, stride)
        print(f"PatchEmbedding: In={in_channels}, Out={embed_dim}, Kernel={patch_size}, Stride={stride}, Pad={padding}, In_L={ppg_length} -> Num Patches={self.num_patches}")

    def _get_output_length(self, L_in, padding, kernel_size, stride):
         L_out = math.floor((L_in + 2 * padding - kernel_size) / stride) + 1
         return L_out

    def forward(self, x):
        x = self.proj(x)
        if x.shape[2] != self.num_patches:
             print(f"Warning: Patch output length mismatch! Expected {self.num_patches}, got {x.shape[2]}.")
        x = x.transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU, dropout_rate=0.1):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ConditionalLayerNorm(nn.Module):
    def __init__(self, normalized_shape, condition_dim):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.condition_dim = condition_dim
        self.gamma_generator = nn.Linear(condition_dim, self.normalized_shape[-1])
        self.beta_generator = nn.Linear(condition_dim, self.normalized_shape[-1])
        self.layer_norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        nn.init.ones_(self.gamma_generator.weight)
        nn.init.zeros_(self.gamma_generator.bias)
        nn.init.zeros_(self.beta_generator.weight)
        nn.init.zeros_(self.beta_generator.bias)

    def forward(self, x, condition_vec):
        gamma = self.gamma_generator(condition_vec)
        beta = self.beta_generator(condition_vec)
        view_shape = [x.size(0)] + [1] * (x.dim() - 2) + [-1]
        gamma = gamma.view(*view_shape)
        beta = beta.view(*view_shape)
        normalized_x = self.layer_norm(x)
        return normalized_x * gamma + beta

# ==================================================
# 1. 核心CDN模型组件 (适配后 V2)
# ==================================================
class FactorEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, condition_dim=32, dropout_rate=0.1): # Reduced complexity
        super().__init__()
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim if condition_dim and condition_dim > 0 else None
        encoder_layers = []
        for _ in range(num_layers):
            use_conditional = self.condition_dim is not None
            norm1 = ConditionalLayerNorm(embed_dim, self.condition_dim) if use_conditional else nn.LayerNorm(embed_dim)
            norm2 = ConditionalLayerNorm(embed_dim, self.condition_dim) if use_conditional else nn.LayerNorm(embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
                dropout=dropout_rate, activation=F.gelu, batch_first=True, norm_first=True
            )
            if use_conditional:
                encoder_layer.norm1 = norm1
                encoder_layer.norm2 = norm2
            encoder_layers.append(encoder_layer)
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.final_norm = ConditionalLayerNorm(embed_dim, self.condition_dim) if self.condition_dim else nn.LayerNorm(embed_dim)

    def forward(self, src, condition_vec=None):
        output = src
        for layer in self.transformer_encoder:
            if self.condition_dim and isinstance(layer.norm1, ConditionalLayerNorm):
                 res = output
                 x = layer.norm1(output, condition_vec)
                 attn_output, _ = layer.self_attn(x, x, x, need_weights=False)
                 output = res + layer.dropout1(attn_output)
                 res = output
                 x = layer.norm2(output, condition_vec)
                 ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                 output = res + layer.dropout2(ff_output)
            else:
                 output = layer(output)
        output = self.final_norm(output, condition_vec) if self.condition_dim else self.final_norm(output)
        mixed_latent = output.mean(dim=1)
        return mixed_latent

class LatentFactorProjectors(nn.Module):
    def __init__(self, input_dim, factor_dims, num_factors, condition_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.factor_dims = factor_dims
        self.num_factors = num_factors
        self.condition_dim = condition_dim if condition_dim and condition_dim > 0 else None
        shared_hidden_dim = input_dim // 2
        mlp_input_dim = input_dim + (self.condition_dim if self.condition_dim else 0)
        self.shared_mlp = MLP(mlp_input_dim, [shared_hidden_dim], shared_hidden_dim)
        self.factor_heads = nn.ModuleList([
            nn.Linear(shared_hidden_dim, factor_dims) for _ in range(num_factors)
        ])

    def forward(self, mixed_latent, condition_vec=None):
        if self.condition_dim and condition_vec is not None:
            combined_input = torch.cat([mixed_latent, condition_vec], dim=1)
        else:
            combined_input = mixed_latent
        shared_features = self.shared_mlp(combined_input)
        z_factors = [self.factor_heads[i](shared_features) for i in range(self.num_factors)]
        return z_factors

class GlucoseEstimator(nn.Module):
    def __init__(self, glucose_factor_dim, hidden_dims=[32, 16]):
        super().__init__()
        self.estimator = MLP(glucose_factor_dim, hidden_dims, output_dim=1, activation=nn.ReLU, dropout_rate=0.1)

    def forward(self, z_glucose):
        return self.estimator(z_glucose)

class SignalReconstructor(nn.Module):
    def __init__(self, total_factor_dim, ppg_channels, ppg_length, hidden_dims=[128, 256]):
        super().__init__()
        self.ppg_channels = ppg_channels
        self.ppg_length = ppg_length
        output_dim = ppg_channels * ppg_length
        self.decoder_mlp = MLP(total_factor_dim, hidden_dims, output_dim, activation=nn.ReLU)

    def forward(self, z_all):
        flat_reconstructed_ppg = self.decoder_mlp(z_all)
        return flat_reconstructed_ppg.view(-1, self.ppg_channels, self.ppg_length)

# ==================================================
# 2. CDN 主模型架构 (适配后 V2)
# ==================================================
class CausalDisentanglementNetworkAdaptedV2(nn.Module):
    """ CDN V2: Uses PPG (1, 64) and Context (8 features) """
    def __init__(self,
                 ppg_channels=1, ppg_length=64, context_dim=8, # Updated inputs
                 # Model hyperparameters (tuned for smaller input)
                 patch_size=8, patch_stride=4, embed_dim=64,
                 transformer_heads=4, transformer_layers=2, transformer_dropout=0.1,
                 num_factors=5, factor_dims=16,
                 glucose_factor_index=3,
                 condition_dim=32, # Output dim of context encoder
                 # Sub-module hidden dims
                 reconstructor_hidden_dims=[128, 256],
                 estimator_hidden_dims=[32, 16]
                 ):
        super().__init__()
        print("--- Initializing CausalDisentanglementNetworkAdaptedV2 ---")
        self.ppg_channels = ppg_channels
        self.ppg_length = ppg_length
        self.context_dim = context_dim
        self.num_factors = num_factors
        self.factor_dims = factor_dims
        self.total_factor_dim = num_factors * factor_dims
        self.glucose_factor_index = glucose_factor_index
        self.condition_dim = condition_dim
        if not (0 <= glucose_factor_index < num_factors):
            raise ValueError("glucose_factor_index out of bounds")

        # --- 1. Context Encoder ---
        self.context_encoder = MLP(context_dim, [condition_dim * 2], condition_dim) # Simple MLP
        print(f"Context Encoder: Input={context_dim}, Output={condition_dim}")

        # --- 2. PPG Patch Embedding ---
        self.patch_embed = PatchEmbedding(ppg_channels, embed_dim, patch_size, patch_stride, ppg_length)
        num_patches = self.patch_embed.num_patches
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=num_patches + 5)
        print(f"PPG Embedding: Embed Dim={embed_dim}, Num Patches={num_patches}")

        # --- 3. Factor Encoder (Transformer) ---
        self.factor_encoder = FactorEncoder(embed_dim, transformer_heads, transformer_layers,
                                            condition_dim=self.condition_dim,
                                            dropout_rate=transformer_dropout)
        print(f"Factor Encoder: Layers={transformer_layers}, Heads={transformer_heads}, Conditioned={self.condition_dim is not None}")

        # --- 4. Latent Factor Projectors ---
        self.latent_projectors = LatentFactorProjectors(embed_dim, factor_dims, num_factors,
                                                        condition_dim=self.condition_dim)
        print(f"Latent Projectors: Factors={num_factors}, Factor Dim={factor_dims}")

        # --- 5. Glucose Estimator ---
        self.glucose_estimator = GlucoseEstimator(factor_dims, hidden_dims=estimator_hidden_dims)
        print(f"Glucose Estimator: Input Dim={factor_dims}")

        # --- 6. Signal Reconstructor ---
        # --- 6. Signal Reconstructor ---
        self.signal_reconstructor = SignalReconstructor(self.total_factor_dim, ppg_channels, ppg_length,
                                                        hidden_dims=reconstructor_hidden_dims)  # <--- 修改这里
        print(f"Signal Reconstructor: Input Dim={self.total_factor_dim}")
        print("--- Model Initialized ---")

    def forward(self, x_ppg, x_context): # Updated inputs
        # 1. Encode context
        condition_vec = self.context_encoder(x_context)

        # 2. Embed PPG
        ppg_patches = self.patch_embed(x_ppg)
        ppg_tokens = self.pos_encoding(ppg_patches)

        # 3. Encode PPG factors (conditioned)
        mixed_latent = self.factor_encoder(ppg_tokens, condition_vec)

        # 4. Project latent factors
        z_factors = self.latent_projectors(mixed_latent, condition_vec)

        # 5. Estimate glucose
        z_glucose = z_factors[self.glucose_factor_index]
        predicted_glucose = self.glucose_estimator(z_glucose)

        # 6. Reconstruct signal
        z_all = torch.cat(z_factors, dim=1)
        reconstructed_ppg = self.signal_reconstructor(z_all)

        return {
            "predicted_glucose": predicted_glucose,
            "reconstructed_ppg": reconstructed_ppg,
            "latent_factors": z_factors,
            "condition_vec": condition_vec
        }

# ==================================================
# CDN 损失函数 (来自你的原始代码, 稍作调整)
# ==================================================
class CDNLossAdapted(nn.Module):
    def __init__(self, weight_pred=1.0, weight_recon=0.5, weight_disentangle=0.1, num_factors=5, factor_dims=16):
        super().__init__()
        self.weight_pred = weight_pred
        self.weight_recon = weight_recon
        self.weight_disentangle = weight_disentangle
        self.num_factors = num_factors
        self.factor_dims = factor_dims
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss() # Use L1 for prediction

    def _calculate_disentanglement_loss(self, z_factors):
        batch_size = z_factors[0].size(0)
        if batch_size <= 1: return torch.tensor(0.0, device=z_factors[0].device)
        z_all = torch.cat(z_factors, dim=1)
        z_centered = z_all - z_all.mean(dim=0, keepdim=True)
        cov_matrix = torch.matmul(z_centered.T, z_centered) / (batch_size - 1)
        total_factor_dim = self.num_factors * self.factor_dims
        if total_factor_dim <= 1: return torch.tensor(0.0, device=z_factors[0].device)
        off_diag_mask = ~torch.eye(total_factor_dim, dtype=torch.bool, device=cov_matrix.device)
        # Use mean absolute value of off-diagonal elements
        loss_disentangle = torch.mean(torch.abs(cov_matrix[off_diag_mask]))
        return loss_disentangle

    def forward(self, model_outputs, ground_truth_glucose, original_ppg):
        loss_pred = self.l1_loss(model_outputs["predicted_glucose"], ground_truth_glucose)
        loss_recon = self.mse_loss(model_outputs["reconstructed_ppg"], original_ppg)
        loss_disentangle = self._calculate_disentanglement_loss(model_outputs["latent_factors"])
        total_loss = (self.weight_pred * loss_pred +
                      self.weight_recon * loss_recon +
                      self.weight_disentangle * loss_disentangle)
        return total_loss, loss_pred, loss_recon, loss_disentangle

# --- 使用示例 ---
if __name__ == '__main__':
    print("Testing Adapted CDN Model V2...")
    bs = 4
    ppg_len = 64
    context_dim = 8 # Demo(4) + Vitals(4)
    model = CausalDisentanglementNetworkAdaptedV2(ppg_length=ppg_len, context_dim=context_dim)

    dummy_ppg = torch.randn(bs, 1, ppg_len)
    dummy_context = torch.randn(bs, context_dim)
    dummy_target = torch.rand(bs, 1) * 10 + 5 # mmol/L range

    try:
        outputs = model(dummy_ppg, dummy_context)
        print("Forward pass successful.")
        print("Predicted Glucose shape:", outputs['predicted_glucose'].shape)

        criterion = CDNLossAdapted(num_factors=model.num_factors, factor_dims=model.factor_dims)
        loss, lp, lr, ld = criterion(outputs, dummy_target, dummy_ppg)
        print("Loss calculation successful:")
        print(f"  Total Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"\nError during V2 test: {e}")
        import traceback
        traceback.print_exc()