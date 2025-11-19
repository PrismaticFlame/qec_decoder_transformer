import torch
import torch.nn as nn
from .embeddings import StabilizerEmbedding


class ConvFeatureExtractor(nn.Module):
    """
    1D convolutional feature extractor over the stabilizer sequence.
    Approximates the 'Convolution layers' + 'Dense block widening' part
    of AlphaQubit's syndrome transformer.
    """

    def __init__(self, d_model: int, conv_dim: int = 128, num_layers: int = 3, widen_factor: int = 5):
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
            x_c = out + residual  # ResNet-style

        x = x_c.transpose(1, 2)  # back to (B, L, d_model)
        return self.norm(x)


class ReadoutResNet(nn.Module):
    """
    Readout head: small ResNet MLP over pooled transformer state.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()

        layers = []
        in_dim = d_model
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 1)  # logical flip logit

    def forward(self, x):
        """
        x: (B, d_model)
        """
        h = self.net(x)
        return self.out(h)


class QECAlphaTransformer(nn.Module):
    """
    AlphaQubit-style QEC transformer (hard-input, simplified):

    - StabilizerEmbedding (Fig. 1C)
    - Conv feature extractor (3 layers)
    - Transformer encoder (3 layers, d_model=256, heads=4)
    - Readout ResNet head
    """

    def __init__(
        self,
        num_stab: int,
        num_cycles: int,
        d_model: int = 256,
        nhead: int = 4,
        num_transformer_layers: int = 3,
        conv_dim: int = 128,
        conv_layers: int = 3,
        conv_widen: int = 5,
        readout_hidden: int = 64,
        readout_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = StabilizerEmbedding(num_stab, num_cycles, d_model)

        self.conv_extractor = ConvFeatureExtractor(
            d_model=d_model,
            conv_dim=conv_dim,
            num_layers=conv_layers,
            widen_factor=conv_widen,
        )

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
            syndrome: (B, L) 0/1
            stab_id:  (L,)
            cycle_id: (L,)

        Returns:
            logit: (B, 1)  logical flip logits
        """
        # 1) stabilizer embedding
        x = self.embedding(syndrome, stab_id, cycle_id)  # (B, L, d_model)

        # 2) local conv feature extractor
        x = self.conv_extractor(x)  # (B, L, d_model)

        # 3) global transformer
        x = self.transformer_encoder(x)  # (B, L, d_model)

        # 4) pool + readout
        pooled = x.mean(dim=1)  # (B, d_model)
        logit = self.readout(pooled)  # (B, 1)
        return logit
