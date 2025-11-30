# text-driven state space model for spatial temporal fusion (S4AMSTF)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from typing import Callable
from timm.layers import DropPath, trunc_normal_
from mamba_ssm import selective_scan_fn
from text_transformer_encoder import TextTransformerEncoder, TextWeightTransformerEncoder
from text_dynamic_graph_encoder import DynamicGraphConv
from text_feature_from_images import ExtractTextFeaturesFromTensors, from_key_words_to_embedding, extract_text_feature


class PatchEmbed2D(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=6, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
    
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
    
    
class SSMBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0.,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            ssm_drop_rate: float = 0,
            d_state: int = 16,
    ):
        super().__init__()
        self.ln1 = norm_layer(hidden_dim)
        self.ln2 = norm_layer(hidden_dim)
        self.ss2d = SS2D(d_model=hidden_dim, dropout=ssm_drop_rate, d_state=d_state)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path1(self.ss2d(self.ln1(x)))
        x = x + self.drop_path2(self.ffn(self.ln2(x)))
        return x


class Patches2Images(nn.Module):    
    def forward(self, x):
        return torch.permute(x, dims=(0, 3, 1, 2))


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )
        
        
class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
    
    
class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        _, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        _, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        # out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
        #       self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        out = self.attn_x(attn) + self.attn_y(attn)

        out = out + local
        # out = self.pad_out(out)
        # out = self.proj(out)
        # out = out[:, :, :H, :W]

        return out
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3001, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3001, dropout=0.1):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.dropout(x)
      

def chw2hwc(x: torch.Tensor):
    return x.permute(0, 2, 3, 1)


def hwc2chw(x: torch.tensor):
    return x.permute(0, 3, 1, 2)


def hw_c2h_w_c(x: torch.Tensor, h):
    b, hw, c = x.shape
    w = hw // h
    return x.reshape(b, h, w, c)


# semantic state space attention model for spatial temporal fusion
class S3AMSTF(nn.Module):
    def __init__(self, in_c=6, n_layer=4,
                 patch_size=2, hidden_dim=96, dpr=0.1,
                 num_heads=8, window_size=5,
                 text_feature_dim_in=768,
                 ):
        super().__init__()
        self.n_layer = n_layer
        self.hc = hidden_dim
        self.ps = patch_size
        
        # modis
        self.proj_m = PatchEmbed2D(patch_size, in_c * 2, hidden_dim, nn.LayerNorm)
        self.path_m = []
        self.fuse_m = []
        self.pe_m = []
        for _ in range(self.n_layer):
            self.path_m.append(SSMBlock(hidden_dim, dpr))
            self.fuse_m.append(nn.Sequential(
                nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4),
                ))
            self.pe_m.append(LearnedPositionalEncoding(hidden_dim))
        self.path_m = nn.ModuleList(self.path_m)
        self.fuse_m = nn.ModuleList(self.fuse_m)
        self.pe_m = nn.ModuleList(self.pe_m)
        
        # landsat
        self.proj_l = PatchEmbed2D(patch_size, in_c, hidden_dim, nn.LayerNorm)  # b, c, h, w -> b, h//patch_size, w//patch_size, hidden_dim
        self.path_l = []
        self.fuse_l = []
        self.pe_l = []
        for _ in range(self.n_layer):
            self.path_l.append(SSMBlock(hidden_dim, dpr))
            self.fuse_l.append(nn.Sequential(
                nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4),
                ))
            self.pe_l.append(LearnedPositionalEncoding(hidden_dim))
        self.path_l = nn.ModuleList(self.path_l)
        self.fuse_l = nn.ModuleList(self.fuse_l)
        self.pe_l = nn.ModuleList(self.pe_l)
        
        # text features to weight visual features
        self.proj_txt_m = nn.Sequential(
            nn.Linear(text_feature_dim_in, hidden_dim),
            nn.Sigmoid(),)
        self.proj_txt_l = nn.Sequential(
            nn.Linear(text_feature_dim_in, hidden_dim),
            nn.Sigmoid(),)
        
        # fuse layer
        self.fuse = []
        for _ in range(self.n_layer):
            self.fuse.append(GlobalLocalAttention(hidden_dim * 2, num_heads=num_heads, window_size=window_size))
        self.fuse = nn.ModuleList(self.fuse)
        
        # out layer
        self.out = nn.Sequential(
            Patches2Images(),
            nn.UpsamplingBilinear2d(scale_factor=patch_size),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, in_c, 3, 1, 1),
        )

    def forward(self, x1, x2, x3, xt):
        xm = torch.cat((x1, x2), dim=1)
        feat_m = self.proj_m(xm)
        feat_l = self.proj_l(x3)
        
        feat_tm = self.proj_txt_m(xt)
        feat_tl = self.proj_txt_l(xt)
        
        for idx in range(self.n_layer):
            feat_m = self.path_m[idx](feat_m)
            _, hm, wm, _ = feat_m.shape
            feat_m = feat_m.view(feat_m.size(0), -1, feat_m.size(-1))
            feat_m = torch.cat((feat_tm.view(feat_tm.size(0), 1, feat_tm.size(-1)), feat_m), dim=1)
            feat_m_tmp = self.fuse_m[idx](feat_m)
            feat_m, feat_tm = feat_m_tmp[:, 1:hm * wm + 1, :], feat_m_tmp[:, 0, :]
            feat_m = feat_m.reshape(feat_m.size(0), hm, wm, feat_m.size(-1))
            
            feat_l = self.path_l[idx](feat_l)            
            _, hl, wl, _ = feat_l.shape
            feat_l = feat_l.view(feat_l.size(0), -1, feat_l.size(-1))
            feat_l = torch.cat((feat_tl.view(feat_tl.size(0), 1, feat_tl.size(-1)), feat_l), dim=1)
            feat_l_tmp = self.fuse_l[idx](feat_l)
            feat_l, feat_tl = feat_l_tmp[:, 1:hl * wl + 1, :], feat_l_tmp[:, 0, :]
            feat_l = feat_l.reshape(feat_l.size(0), hl, wl, feat_l.size(-1))
            
            feat_fuse = torch.cat((hwc2chw(feat_l), hwc2chw(feat_m)), dim=1)
            feat_fuse = self.fuse[idx](feat_fuse)
            feat_fuse = chw2hwc(feat_fuse)
            feat_l, feat_m = feat_fuse[:, :, :, 0: self.hc], feat_fuse[:, :, :, self.hc: self.hc * 2]
        out = self.out(feat_l)
        return out        


if '__main__' == __name__:
    bs = 2
    ps = 100
    aa = torch.randint(low=-100, high=1000, size=(bs, 6, ps, ps)).cuda().to(dtype=torch.int16)
    bb = torch.randint(low=-100, high=1000, size=(bs, 6, ps, ps)).cuda().to(dtype=torch.int16)
    cc = torch.randint(low=-100, high=1000, size=(bs, 6, ps, ps)).cuda().to(dtype=torch.int16)
    tt = torch.rand(size=(bs, 768)).cuda()
    mm = S3AMSTF().cuda()
    dd = mm(aa.to(dtype=torch.float32), bb.to(dtype=torch.float32), cc.to(dtype=torch.float32), tt)
    print(dd.shape)
    torch.save(mm.state_dict(), 'model_size_test.pth')
    
