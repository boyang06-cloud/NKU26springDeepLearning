
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, use_dw=True):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        groups = dim * mlp_ratio if use_dw else 1
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=groups)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.att = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.att(x) * self.v(x)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, index, dim, kernel_size, num_head, window_size=14, mlp_ratio=4., drop_path=0., use_dw=True):
        super().__init__()
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(dim, mlp_ratio, use_dw=use_dw)
        self.layer_scale_1 = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Conv2Former(nn.Module):
    def __init__(self, kernel_size=None, kernel_sizes=None, img_size=224, in_chans=3, num_classes=1000,
                 depths=None, dims=None, window_sizes=None,
                 mlp_ratios=None, num_heads=None,
                 drop_path_rate=0., head_dim=1280, mlp_use_dw=None):
        super().__init__()
        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 768]
        if window_sizes is None:
            window_sizes = [14, 14, 14, 7]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if num_heads is None:
            num_heads = [2, 4, 10, 16]

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(dims)

        if mlp_use_dw is None:
            mlp_use_dw = [True] * self.num_stages
        assert len(mlp_use_dw) == self.num_stages, f"mlp_use_dw length {len(mlp_use_dw)} != num_stages {self.num_stages}" 

        if kernel_sizes is None:
            k = kernel_size if kernel_size is not None else 7
            kernel_sizes = [k] * self.num_stages
        assert len(kernel_sizes) == self.num_stages

        self.downsample_layers = nn.ModuleList()
        if img_size < 56:
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]),
                nn.GELU(),
            )
        else:
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GELU(),
                nn.BatchNorm2d(dims[0] // 2),
                nn.Conv2d(dims[0] // 2, dims[0] // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GELU(),
                nn.BatchNorm2d(dims[0] // 2),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=2, stride=2, bias=False),
            )
        self.downsample_layers.append(stem)

        for i in range(self.num_stages - 1):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(*[
                Block(index=cur + j, dim=dims[i], kernel_size=kernel_sizes[i],
                      drop_path=dp_rates[cur + j], num_head=num_heads[i],
                      window_size=window_sizes[i], mlp_ratio=mlp_ratios[i],
                      use_dw=mlp_use_dw[i])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.head = nn.Sequential(
            nn.Conv2d(dims[-1], head_dim, 1),
            nn.GELU(),
            LayerNorm(head_dim, eps=1e-6, data_format="channels_first")
        )
        self.pred = nn.Linear(head_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.head(x)
        return x.mean([-2, -1])

    def forward(self, x):
        return self.pred(self.forward_features(x))




# --- CIFAR-100 variants ---

@register_model
def conv2former_cifar100_t(pretrained=False, **kwargs):
    drop_path_rate = kwargs.pop('drop_path_rate', 0.05)
    model = Conv2Former(
        kernel_sizes=[5, 5, 3, 3], img_size=28, num_classes=100,
        dims=[40, 80, 160, 320], mlp_ratios=[2, 2, 3, 3],
        depths=[1, 2, 4, 2], num_heads=[2, 4, 8, 16],
        mlp_use_dw=[False, False, True, True],
        window_sizes=[14, 14, 7, 4], drop_path_rate=drop_path_rate, head_dim=256, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def conv2former_cifar100_s(pretrained=False, **kwargs):
    drop_path_rate = kwargs.pop('drop_path_rate', 0.1)
    model = Conv2Former(
        kernel_sizes=[5, 5, 3, 3], img_size=28, num_classes=100,
        dims=[48, 96, 192, 384], mlp_ratios=[2, 2, 4, 4],
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        mlp_use_dw=[False, False, True, True],
        window_sizes=[14, 14, 7, 4], drop_path_rate=drop_path_rate, head_dim=512, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def conv2former_cifar100_b(pretrained=False, **kwargs):
    drop_path_rate = kwargs.pop('drop_path_rate', 0.15)
    model = Conv2Former(
        kernel_sizes=[7, 5, 3, 3], img_size=28, num_classes=100,
        dims=[64, 128, 256, 512], mlp_ratios=[2, 2, 4, 4],
        depths=[2, 3, 8, 2], num_heads=[4, 8, 16, 32],
        mlp_use_dw=[False, False, True, True],
        window_sizes=[14, 14, 7, 4], drop_path_rate=drop_path_rate, head_dim=768, **kwargs)
    model.default_cfg = _cfg()
    return model
