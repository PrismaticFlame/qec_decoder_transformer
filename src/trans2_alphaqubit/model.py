import torch
import torch.nn as nn
from .embeddings import StabilizerEmbedding


class ConvFeatureExtractor(nn.Module):
    """
    Scaling 規格：
    - Convolution layers: 3
    - Convolution dimensions: 128
    - Dense block dimension widening: 5
    """

    def __init__(
        self,
        d_model: int,
        conv_dim: int = 128,        # Scaling: 128
        num_layers: int = 3,
        widen_factor: int = 5,
    ):
        super().__init__()

        layers = []
        in_channels = d_model
        hidden = conv_dim
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden, hidden * widen_factor, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden * widen_factor, d_model, kernel_size=1),
                )
            )
            in_channels = d_model

        self.blocks = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, L, d_model)
        """
        x_c = x.transpose(1, 2)  # (B, d_model, L)

        for block in self.blocks:
            residual = x_c
            out = block(x_c)
            x_c = out + residual  # ResNet-style skip

        x = x_c.transpose(1, 2)  # back to (B, L, d_model)
        return self.norm(x)


class ReadoutResNet(nn.Module):
    """
    Readout head: Scaling 規格

    - Layers: 16
    - Dimensions: 48
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 48,   # Scaling: 48
        num_layers: int = 16,   # 16
    ):
        super().__init__()

        blocks = []
        in_dim = d_model
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            blocks.append(block)
            in_dim = hidden_dim

        self.blocks = nn.ModuleList(blocks)
        self.final_act = nn.ReLU(inplace=True)
        self.out = nn.Linear(hidden_dim, 1)  # logical flip logit

    def forward(self, x):
        """
        x: (B, d_model)
        """
        h = x
        for block in self.blocks:
            residual = h
            h = block(h)
            h = self.final_act(h + residual)
        return self.out(h)


class QECAlphaTransformer(nn.Module):
    """
    AlphaQubit-style QEC transformer （Scaling 規格）:

    - StabilizerEmbedding + 2-layer ResNet feature embedding
    - Conv feature extractor (3 layers, dim 128, widen 5)
    - Transformer encoder (3 layers, d_model=256, heads=4)
    - Readout ResNet head (16 layers, dim 48)
    """

    def __init__(
        self,
        num_stab: int,
        num_cycles: int,
        distance: int = 3,
        # ---- Syndrome transformer hyperparams: Scaling ----
        d_model: int = 256,          # Scaling: 256
        nhead: int = 4,
        num_transformer_layers: int = 3,
        # ---- Conv feature extractor ----
        conv_dim: int = 128,         # Scaling: 128
        conv_layers: int = 3,
        conv_widen: int = 5,
        # ---- Readout head ----
        readout_hidden: int = 48,    # Scaling: 48
        readout_layers: int = 16,    # 16
        # ---- Generic ----
        dropout: float = 0.1,
        embedding_resnet_layers: int = 2,  # Feature embedding ResNet layers
    ):
        super().__init__()

        # 1) stabilizer embedding
        self.embedding = StabilizerEmbedding(num_stab, num_cycles, d_model)

        # 1b) Feature embedding ResNet (2 層)
        emb_blocks = []
        for _ in range(embedding_resnet_layers):
            emb_blocks.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(inplace=True),
                    nn.Linear(d_model, d_model),
                )
            )
        self.embedding_resnet = nn.ModuleList(emb_blocks)
        self.embedding_norm = nn.LayerNorm(d_model)

        # 2) local conv feature extractor
        self.conv_extractor = ConvFeatureExtractor(
            d_model=d_model,
            conv_dim=conv_dim,
            num_layers=conv_layers,
            widen_factor=conv_widen,
        )

        # 3) global transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # 4) readout ResNet
        self.readout = ReadoutResNet(
            d_model=d_model,
            hidden_dim=readout_hidden,
            num_layers=readout_layers,
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, syndrome, stab_id, cycle_id):
        """
        Args:
            syndrome: (B, L)
            stab_id:  (L,)
            cycle_id: (L,)

        Returns:
            logit: (B, 1)
        """
        # 1) stabilizer embedding
        x = self.embedding(syndrome, stab_id, cycle_id)  # (B, L, d_model)

        # 1b) Feature embedding ResNet
        B, L, D = x.shape
        h = x.view(B * L, D)
        for block in self.embedding_resnet:
            residual = h
            h = block(h)
            h = h + residual
        h = self.embedding_norm(h)
        x = h.view(B, L, D)

        # 2) local conv feature extractor
        x = self.conv_extractor(x)  # (B, L, d_model)

        # 3) global transformer
        x = self.transformer_encoder(x)  # (B, L, d_model)

        # 4) pool + readout
        pooled = x.mean(dim=1)  # (B, d_model)
        logit = self.readout(pooled)  # (B, 1)
        return logit
