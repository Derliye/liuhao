import torch
import torch.nn as nn
from einops import rearrange
from net.transformer_utils import *

class CDEM(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CDEM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.linear = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.delta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.nn.functional.softmax(attn,dim=-1)

        z_t = (attn @ v)
        z_t = rearrange(z_t, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        t_prime = self.linear(z_t)  # T'_T
        t_1 = self.beta * t_prime + self.alpha * y

        ffn_out = self.ffn(t_1)
        t_2 = self.gamma * t_1 + self.delta * ffn_out

        t_2 = self.project_out(t_2)
        return t_2
    

class MFEM(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(MFEM, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias),    
            nn.ReLU(inplace=True),   
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),           
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(1, 3), padding=(0, 1), groups=hidden_features, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 1), padding=(1, 0), groups=hidden_features, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=3, dilation=3, groups=hidden_features, bias=bias),
            nn.ReLU(inplace=True),
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),           
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 1), padding=(1, 0), groups=hidden_features, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(1, 3), padding=(0, 1), groups=hidden_features, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=3, dilation=3, groups=hidden_features, bias=bias),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
        )
 
        self.conv_cat = nn.Conv2d(hidden_features * 4, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv_cat(x_cat)
        return out


class CDEM_HV(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(CDEM_HV, self).__init__()
        self.gdfn = MFEM(dim)
        self.norm = LayerNorm(dim)
        self.ffn = CDEM(dim, num_heads, bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x
    
class CDEM_I(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(CDEM_I, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = MFEM(dim)
        self.ffn = CDEM(dim, num_heads, bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x)) 
        return x
