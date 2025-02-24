import torch
from torch import nn

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5),
            nn.MaxPool1d(2),
            nn.Conv1d(4, 8, 5),
            nn.MaxPool1d(2),
            nn.ReLU()
        )
        # self.transformer_encoder = nn.TransformerEncoderLayer(182, 2, dim_feedforward=256, batch_first=True)
        # self.lstm = nn.LSTM(input_size=182, hidden_size=256, batch_first=True)
        # self.attention = nn.MultiheadAttention(256, 2, batch_first=True)

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 182, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = x.float()
        x = self.cnn(x.unsqueeze(1))
        # x = self.transformer_encoder(x)
        # x, _ = self.lstm(x)
        # x, _ = self.attention(x, x, x)
        x = self.linear(x)
        return x

    def train_pre(self, x):
        with torch.no_grad():
            x = self.forward(x)
        return self.mlp(x)