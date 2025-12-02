#!/usr/bin/python3

import torch
import torch.nn as nn


class PoseTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        # Projecting dimensions in (V*2) to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional embeddings
        self.max_len = 1 + 256
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_len, d_model))

        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward_features(self, x):
        B, T, _ = x.shape

        x = self.input_proj(x)

        cls = self.cls_token.expand(B, 1, -1)

        x = torch.cat([cls, x], dim=1)

        pos = self.pos_embedding[:, : 1 + T, :]
        x = x + pos

        x = self.encoder(x)
        cls_out = x[:, 0, :]
        return cls_out

    def forward(self, x):
        cls_out = self.forward_features(x)
        cls_out = self.dropout(cls_out)
        logits = self.fc(cls_out)
        return logits
