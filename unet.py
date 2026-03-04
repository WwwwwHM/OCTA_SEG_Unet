import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3_12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv3_18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.out_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3_6(x)
        x3 = self.conv3_12(x)
        x4 = self.conv3_18(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.out_conv(x_cat)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        attn = self.sigmoid(attn)
        return x * attn


class ECAAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class PDEAttention(nn.Module):
    def __init__(self, channels):
        super(PDEAttention, self).__init__()
        self.grad_x = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.grad_y = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.weight_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gx = self.grad_x(x)
        gy = self.grad_y(x)
        grad = torch.cat([gx, gy], dim=1)
        attn = self.sigmoid(self.weight_conv(grad))
        return x * attn


class EdgeBranch(nn.Module):
    def __init__(self, in_channels):
        super(EdgeBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

    def forward(self, x, target_size):
        edge = self.conv(x)
        return F.interpolate(edge, size=target_size, mode='bilinear', align_corners=False)

# 门控注意力模块
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# 简化版Transformer Encoder Block
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super(SimpleTransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, HW, C]
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_flat + attn_out
        x_norm2 = self.norm2(x)
        x = x + self.mlp(x_norm2)
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        return x

class UNet_Transformer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],
        trans_dim=1024,
        use_residual=False,
        use_gated_attention=True,
        use_eca_attention=False,
        use_spatial_attention=False,
        use_pde_attention=False,
        use_aspp=False,
        use_edge_branch=False,
        return_edge=False
    ):
        super(UNet_Transformer, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.gated_attentions = nn.ModuleList()
        self.eca_attentions = nn.ModuleList()
        self.spatial_attentions = nn.ModuleList()
        self.pde_attentions = nn.ModuleList()
        self.use_residual = use_residual
        self.use_gated_attention = use_gated_attention
        self.use_eca_attention = use_eca_attention
        self.use_spatial_attention = use_spatial_attention
        self.use_pde_attention = use_pde_attention
        self.use_aspp = use_aspp
        self.use_edge_branch = use_edge_branch
        self.return_edge = return_edge

        def make_block(in_ch, out_ch):
            return ResidualBlock(in_ch, out_ch) if self.use_residual else ConvBlock(in_ch, out_ch)

        # 下采样部分
        for feature in features:
            self.downs.append(make_block(in_channels, feature))
            in_channels = feature
        # Transformer bottleneck
        self.bottleneck_conv = nn.Conv2d(features[-1], trans_dim, kernel_size=1)
        self.transformer = SimpleTransformerEncoder(trans_dim)
        bottleneck_out_channels = features[-1] * 2
        self.bottleneck_deconv = nn.Conv2d(trans_dim, bottleneck_out_channels, kernel_size=1)
        self.aspp = ASPP(bottleneck_out_channels, bottleneck_out_channels) if self.use_aspp else nn.Identity()
        self.edge_branch = EdgeBranch(bottleneck_out_channels) if self.use_edge_branch else None
        # 上采样部分
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(make_block(feature*2, feature))
            if self.use_gated_attention:
                self.gated_attentions.append(AttentionBlock(F_g=feature, F_l=feature, F_int=feature//2))
            if self.use_eca_attention:
                self.eca_attentions.append(ECAAttention(feature))
            if self.use_spatial_attention:
                self.spatial_attentions.append(SpatialAttention())
            if self.use_pde_attention:
                self.pde_attentions.append(PDEAttention(feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Transformer bottleneck
        x = self.bottleneck_conv(x)
        x = self.transformer(x)
        x = self.bottleneck_deconv(x)
        x = self.aspp(x)
        edge_out = self.edge_branch(x, target_size=input_size) if self.edge_branch is not None else None
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            refined_skip = skip_connection
            if self.use_gated_attention:
                refined_skip = self.gated_attentions[idx//2](g=x, x=refined_skip)
            if self.use_eca_attention:
                refined_skip = self.eca_attentions[idx//2](refined_skip)
            if self.use_spatial_attention:
                refined_skip = self.spatial_attentions[idx//2](refined_skip)
            if self.use_pde_attention:
                refined_skip = self.pde_attentions[idx//2](refined_skip)

            if x.shape != refined_skip.shape:
                x = F.interpolate(x, size=refined_skip.shape[2:])
            x = torch.cat((refined_skip, x), dim=1)
            x = self.ups[idx+1](x)

        seg_out = self.final_conv(x)
        if self.return_edge and edge_out is not None:
            return seg_out, edge_out
        return seg_out
