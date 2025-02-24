import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float32)

class Self_Attn(nn.Module):
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1), nn.ReLU())

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1), nn.ReLU(),
                                  DoubleConv(out_channels, out_channels, residual=True),
                                  DoubleConv(out_channels, out_channels, residual=True))
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x, t):
        x = self.conv(x)
        emb = self.emb_layer(t).float()[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, conv1=False):
        super().__init__()
        self.conv1 = conv1
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1), nn.ReLU())

        # self.up = nn.Upsample(scale_factor=(2, 1), mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(DoubleConv(in_channels, in_channels, residual=True),
                                  DoubleConv(in_channels, out_channels, in_channels // 2))

        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x, skip_x, t, conv1=False):
        # x = x.unsqueeze(-1)
        # x = self.up(x).squeeze(-1)
        if conv1:
            x = self.conv1(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb

class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        # self.preprocess = nn.Sequential(nn.Conv2d(c_in, 32, kernel_size=(1, 6), padding=(0, 3)),
        #                                 nn.ReLU(), nn.Conv2d(32, 64, kernel_size=(1, 2)), nn.ReLU())
        self.preprocess = nn.Sequential(nn.Conv1d(c_in, 32, kernel_size=3, padding=1),
                                        nn.ReLU(), nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.time_dim = time_dim
        self.inc = DoubleConv(64, 64)
        self.down1 = Down(64, 128)
        self.sa1 = nn.MultiheadAttention(128, num_heads=4, batch_first=True)  # Self_Attn(128, 'relu')
        self.down2 = Down(128, 256)
        self.sa2 = nn.MultiheadAttention(256, num_heads=4, batch_first=True)  # Self_Attn(256, 'relu')
        self.down3 = Down(256, 256)
        self.sa3 = nn.MultiheadAttention(256, num_heads=4, batch_first=True)  # Self_Attn(256, 'relu')

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = nn.MultiheadAttention(128, num_heads=4, batch_first=True)  # Self_Attn(128, 'relu')
        self.up2 = Up(256, 64)
        self.sa5 = nn.MultiheadAttention(64, num_heads=4, batch_first=True)  # Self_Attn(64, 'relu')
        self.up3 = Up(128, 64)
        self.sa6 = nn.MultiheadAttention(64, num_heads=4, batch_first=True)  # Self_Attn(64, 'relu')
        self.outc = nn.Conv1d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).float()
        t = self.pos_encoding(t, self.time_dim)
        x1 = self.preprocess(x.float())
        x2 = self.inc(x1)
        x3 = self.down1(x2, t).permute(0, 2, 1)
        x3, _ = self.sa1(x3, x3, x3)
        x3 = x3.permute(0, 2, 1)
        x4 = self.down2(x3, t).permute(0, 2, 1)
        x4, _ = self.sa2(x4, x4, x4)
        x4 = x4.permute(0, 2, 1)
        x5 = self.down3(x4, t).permute(0, 2, 1)
        x5, _ = self.sa3(x5, x5, x5)
        x5 = x5.permute(0, 2, 1)

        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up1(x5, x4, t, conv1=True).permute(0, 2, 1)
        x, _ = self.sa4(x, x, x)
        x = x.permute(0, 2, 1)
        x = self.up2(x, x3, t, conv1=True).permute(0, 2, 1)
        x, _ = self.sa5(x, x, x)
        x = x.permute(0, 2, 1)
        x = self.up3(x, x2, t).permute(0, 2, 1)
        x, _ = self.sa6(x, x, x)
        x = x.permute(0, 2, 1)
        output = self.outc(x)
        return output



if __name__ == "__main__":
    nn.TransformerEncoder
