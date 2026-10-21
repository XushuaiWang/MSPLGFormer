import torch
from torch import nn
import torch.nn.functional as F


MODEL_NAME = 'MSPLGFormer'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_factory(config):
    return MSPLGFormer(config, num_classes=config['num_labels'])


class ConvFFN(nn.Module):
    """Lightweight convolutional FFN for token sequences.

    Input/Output shape: [batch, seq_len, d_model].
    The pointwise projections mix channels, while the depthwise temporal
    convolution mixes neighboring positions along the token sequence.
    """

    def __init__(self, d_model, dim_ff, dropout=0.0, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, dim_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim_ff, dim_ff, kernel_size=kernel_size, padding=padding, groups=dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim_ff, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class LightweightTransformerLayer(nn.Module):
    """Pre-norm Transformer encoder layer with ConvFFN.

    Uses multi-head self-attention and a depthwise ConvFFN with residual connections.
    """

    def __init__(self, d_model, num_heads, dim_ff, dropout=0.0, conv_kernel_size=3):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f'd_model_patch ({d_model}) must be divisible by num_heads ({num_heads})')
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.conv_ffn = ConvFFN(
            d_model=d_model,
            dim_ff=dim_ff,
            dropout=dropout,
            kernel_size=conv_kernel_size,
        )

    def forward(self, x):
        attn_input = self.norm_attn(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout_attn(attn_out)
        x = x + self.conv_ffn(self.norm_ffn(x))
        return x


class LightweightTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, num_layers=2, dropout=0.0, conv_kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList([
            LightweightTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_ff=dim_ff,
                dropout=dropout,
                conv_kernel_size=conv_kernel_size,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MultiScalePatchTransformerBlock(nn.Module):
    def __init__(self, seq_len, emb_size, d_model_patch, patch_len, stride, padding_patch,
                 relation_decay, dropout, pap_dropout,
                 transformer_layers, num_heads, dim_ff, conv_ffn_kernel_size):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding_patch = padding_patch
        self.d_model_patch = d_model_patch

        work_len = max(seq_len, self.patch_len)
        patch_num = (work_len - self.patch_len) // self.stride + 1
        if self.padding_patch == 'end':
            patch_num += 1

        self.patch_proj = nn.Linear(patch_num, d_model_patch)
        self.transformer = LightweightTransformerEncoder(
            d_model=d_model_patch,
            num_heads=num_heads,
            dim_ff=dim_ff,
            num_layers=transformer_layers,
            dropout=dropout,
            conv_kernel_size=conv_ffn_kernel_size,
        )
        self.mpnn1 = GraphConvpoolMPNNBlock(
            input_dim=d_model_patch,
            output_dim=d_model_patch,
            time_window_size=self.patch_len,
            decay=relation_decay,
        )
        self.mpnn2 = GraphConvpoolMPNNBlock(
            input_dim=d_model_patch,
            output_dim=d_model_patch,
            time_window_size=self.patch_len,
            decay=relation_decay,
        )
        self.pap = PAP(pap_dim=1, dropout_ratio=pap_dropout)

    def forward(self, x):
        # x: [B, emb_size, L]
        batch_size, emb_size, length = x.shape
        if length < self.patch_len:
            x = F.pad(x, (0, self.patch_len - length), mode='replicate')
        if self.padding_patch == 'end':
            x = F.pad(x, (0, self.stride), mode='replicate')

        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B, E, patch_num, patch_len] -> [B*E, patch_len, patch_num]
        patches = patches.reshape(batch_size * emb_size, patches.size(2), patches.size(3)).permute(0, 2, 1)
        tokens = self.patch_proj(patches)
        tokens = self.transformer(tokens)

        mpnn_output1 = self.mpnn1(tokens)
        mpnn_output2 = self.mpnn2(tokens)
        mpnn_pap1 = self.pap(mpnn_output1)
        mpnn_pap2 = self.pap(mpnn_output2)
        features = torch.cat([mpnn_pap1, mpnn_pap2], dim=-1)
        return features.reshape(batch_size, emb_size * features.size(-1))


class MSPLGFormer(nn.Module):
    """MSPLGFormer for multivariate time series classification.

    Architecture:
    Conv2D embedding -> multi-scale patch projection -> configurable lightweight
    Transformer encoder with ConvFFN -> parallel MPNNs -> PAP -> scale fusion
    -> linear classifier. The encoder depth is set by config['transformer_layers'].
    """

    def __init__(self, config, num_classes):
        super().__init__()
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        d_model_patch = config['d_model_patch']
        patch_lens = config['patch_lens']
        strides = config['strides']
        if len(patch_lens) != len(strides):
            raise ValueError('--patch_lens and --strides must have the same number of values')

        self.embed_layer = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(1, emb_size * 4, kernel_size=(1, 8), padding=0),
            nn.BatchNorm2d(emb_size * 4),
            nn.GELU(),
        )
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=(channel_size, 1), padding=0),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
        )

        self.scale_blocks = nn.ModuleList([
            MultiScalePatchTransformerBlock(
                seq_len=seq_len,
                emb_size=emb_size,
                d_model_patch=d_model_patch,
                patch_len=patch_len,
                stride=stride,
                padding_patch=config['padding_patch'],
                relation_decay=config['relation_decay'],
                dropout=config['dropout'],
                pap_dropout=config['pap_dropout'],
                transformer_layers=config['transformer_layers'],
                num_heads=config['num_heads'],
                dim_ff=config['dim_ff'],
                conv_ffn_kernel_size=config['conv_ffn_kernel_size'],
            )
            for patch_len, stride in zip(patch_lens, strides)
        ])

        # Learn global scale weights for softmax-weighted feature fusion.
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scale_blocks)))

        # PAP concatenates mean and max statistics into 2 * d_model_patch features.
        # Concatenating both pooled MPNN branches gives 4 * d_model_patch per embedding channel.
        self.out = nn.Linear(emb_size * 4 * d_model_patch, num_classes)

    def forward(self, x):
        # x: [B, C, L]
        x = x.unsqueeze(1)
        x = self.embed_layer(x)
        x = self.embed_layer2(x).squeeze(2)  # [B, emb_size, L]

        scale_features = torch.stack([block(x) for block in self.scale_blocks], dim=1)
        scale_weights = torch.softmax(self.scale_logits, dim=0).to(dtype=scale_features.dtype, device=scale_features.device)
        features = torch.sum(scale_features * scale_weights.view(1, -1, 1), dim=1)
        return self.out(features)


class DotGraphConstructionWeights(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        batch_size, num_nodes, _ = node_features.size()
        adjacency = torch.bmm(node_features, node_features.transpose(1, 2))
        eye = torch.eye(num_nodes, device=node_features.device, dtype=node_features.dtype).unsqueeze(0)
        adjacency = F.leaky_relu(adjacency - eye * 1e8)
        adjacency = F.softmax(adjacency, dim=-1)
        return adjacency + eye


def temporal_decay_matrix(time_length, decay_rate):
    idx = torch.arange(time_length, dtype=torch.float32)
    distance = torch.abs(idx[:, None] - idx[None, :])
    return decay_rate ** distance


class MPNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, k=1):
        super().__init__()
        self.k = k
        self.theta = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(k)])
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x, adjacency):
        outputs = []
        a_power = adjacency
        for i, linear in enumerate(self.theta):
            if i > 0:
                a_power = torch.bmm(a_power, adjacency)
            outputs.append(linear(torch.bmm(a_power, x)))
        out = sum(outputs)
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        return F.leaky_relu(out)


class GraphConvpoolMPNNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, time_window_size, decay):
        super().__init__()
        self.graph_construction = DotGraphConstructionWeights(input_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.mpnn = MPNNLayer(input_dim, output_dim, k=1)
        self.register_buffer('pre_relation', temporal_decay_matrix(time_window_size, decay))

    def forward(self, x):
        adjacency = self.graph_construction(x)
        adjacency = adjacency * self.pre_relation.to(dtype=x.dtype, device=x.device)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        x_norm = self.bn(x.transpose(1, 2)).transpose(1, 2)
        return self.mpnn(x_norm, adjacency)


class PAP(nn.Module):
    def __init__(self, pap_dim, dropout_ratio):
        super().__init__()
        self.pap_dim = pap_dim
        self.dropout = nn.Dropout(p=dropout_ratio, inplace=False)

    def forward(self, x):
        mean_pool = torch.mean(x, dim=self.pap_dim)
        max_pool = torch.amax(x, dim=self.pap_dim)
        return self.dropout(torch.cat([mean_pool, max_pool], dim=-1))
