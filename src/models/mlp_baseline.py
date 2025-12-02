import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x_mean = x.mean(dim=1)
        logits = self.net(x_mean)
        return logits
