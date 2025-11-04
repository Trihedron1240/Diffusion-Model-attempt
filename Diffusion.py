import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset, Subset
import random
from torch.nn import functional as F
from PIL import Image
import numpy as np
from torch import amp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Transform and dataset
transform = transforms.Compose([transforms.Resize(70), transforms.CenterCrop(64), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
celeba_root = "/kaggle/input/celeba-dataset"
dataset = datasets.ImageFolder(root=celeba_root, transform=transform)

#Dataloader
subset_size = min(10000, len(dataset))
if subset_size == 0:
    raise RuntimeError("Dataset is empty")
batch = 16
all_indices = list(range(len(dataset)))
random_indices = random.sample(all_indices, subset_size)
subset = Subset(dataset, random_indices)
dataloader = DataLoader(subset, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

imgs, _ = next(iter(dataloader))
print(imgs.shape == (batch, 3, 64, 64))

# Cumprod = cumulative product
# NOTE: Using a cosine schedule (Nichol # 1e-4, 2e-2 good for diffusion, start small, end with bigger varience Dhariwal); the old linear-beta note does not apply.

# beta_schedule to define variables for the math to come

def make_betas_schedule_cosine(T, s=0.008, device=None, eps=1e-5): # changed: switched to cosine schedule
    # ᾱ(t) = cos^2(((t/T)+s)/(1+s) * π/2)
    steps = torch.arange(T+1, device=device, dtype=torch.float32)
    t = steps / T
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to 1 at t=0
    # convert ᾱ -> β_t
    alphas = alphas_cumprod[1:] / (alphas_cumprod[:-1] + eps)
    betas = (1 - alphas).clamp(1e-8, 0.999)
    return betas, alphas, alphas_cumprod[1:]

# compute for xt and noise from the formula !!! remember formula 
def q_sample(x0, t, alphas_cumprod):
    noise = torch.randn_like(x0)
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1-a_bar) * noise
    return xt, noise

# dimension, using it for sinusoids (cos sin frequencies), more D = more resolution sort of, however too high D = large compute cost

class time_embed(nn.Module):
    def __init__(self, dim: int = 256, out_dim: int = 256, base: float =10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.out_dim = out_dim
        self.base = base
        self.model = nn.Sequential(
            nn.Linear(self.dim, self.dim*4),
            nn.SiLU(),
            nn.Linear(self.dim*4, self.out_dim),
            nn.SiLU()
        )
    def forward(self, t):
        t = t.float()
        half_dim = self.dim // 2 # // used to make sure its an integer
        k = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        freq_k = 1 / (self.base**(k/(self.dim/2)))
        angle_matrix = t[:, None] * freq_k[None, :]
        sin_ = torch.sin(angle_matrix)
        cos_ = torch.cos(angle_matrix)
        concat = torch.cat([sin_, cos_], dim=-1)
        return self.model(concat)



# ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, temb_dim=256, group_num=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_proj = nn.Linear(temb_dim, self.out_channels)
        self.act = nn.SiLU()

        self.first_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_num, self.out_channels),
            nn.SiLU()
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_num, self.out_channels),
            nn.SiLU()
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb):
        temb_proj = self.temb_proj(temb)[:, :, None, None]  # [N,C,1,1]
        feat = self.first_conv(x)         # [N,C,H,W]
        feat = feat + temb_proj           
        feat = self.act(feat)             
        feat = self.second_conv(feat)
        return feat + self.skip(x)

#
class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, is_last=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(in_ch,  out_ch, temb_dim),   # in_ch -> out_ch
            ResBlock(out_ch, out_ch, temb_dim),   # stay at out_ch
        ])
        self.down = nn.Identity() if is_last else nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, temb):
        for b in self.blocks:
            x = b(x, temb)           # now x has out_ch channels
        skip = x                      # skip also out_ch
        x = self.down(x)              # spatial ↓2 unless last
        return x, skip

# SelfAttention2d
class AttentionBlock(nn.Module):
    def __init__(self, channels: int, group_num: int = 32, num_heads: int = 4, attention_dropout: float = 0.0,
                  proj_dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, 'channels must be divisible by num_heads' # saving time in debugging if values are wrong
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.channels = channels
        self.num_heads = num_heads
        self.dim_heads = channels // num_heads

        self.norm = nn.GroupNorm(group_num, channels)
        # Q = queries, K = keys, V = values
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
    def forward(self, x: torch.Tensor):
        N, C, H, W = x.shape
        x_prenorm = self.norm(x)

        q = self.q(x_prenorm)
        k = self.k(x_prenorm)
        v = self.v(x_prenorm)

        def reshape_heads(t):
            t = t.view(N, self.num_heads, self.dim_heads, H * W)
            return t.permute(0, 1, 3, 2) # split channels by num_heads then reshape [N, C, H, W] -> [N, num_heads, H * W, dim_head]
        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        
        scale = self.dim_heads ** -0.5 # better gradients and not too big numbers
        attention = torch.matmul(q, k.transpose(-2, -1)) * scale # transpose swaps two dimensions / matrix multiplication to get [N, C, HW, HW] from [N, heads, HW, d] [N, heads, d, HW]
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        out = torch.matmul(attention, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(N, C, H, W)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out
    
#Bottleneck Block

class BottleneckBlock(nn.Module):
    def __init__(self, channels: int, temb_dim: int = 256, group_num: int = 32, use_attention: bool = True,
                  num_heads: int = 4, attention_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.res1 = ResBlock(channels, channels, temb_dim, group_num)
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(channels, group_num, num_heads, attention_dropout, proj_dropout)
        self.res2 = ResBlock(channels, channels, temb_dim, group_num)
    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        x = self.res1(x, temb)
        if self.use_attention:
            x = x + self.attention(x)
        x = self.res2(x, temb)
        return x
# Upsample2d
class UpSample2d(nn.Module):
    def __init__(self, in_channels, out_channels = None): # out_channels defaults to in_channels
        super().__init__()
        out_channels = out_channels or in_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') # commonly used in diffusion UNets; upsamples x2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)

# Upsample Block
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, temb_dim,
                 use_attention=False, group_num=32):
        super().__init__()
        self.use_attention = use_attention
        self.upsample = UpSample2d(in_channels, out_channels)
        self.merge = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        self.res1 = ResBlock(in_channels=out_channels, out_channels=out_channels,
                             temb_dim=temb_dim, group_num=group_num)
        self.res2 = ResBlock(in_channels=out_channels, out_channels=out_channels,
                             temb_dim=temb_dim, group_num=group_num)
        if self.use_attention:
            self.attention = AttentionBlock(out_channels)
    def forward(self, x, skip, temb):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]: # Handle off-by-one spatial mismatches before concat
            sh, sw = skip.shape[-2:]
            xh, xw = x.shape[-2:]
            dh, dw = (sh - xh) // 2, (sw - xw) // 2
            skip = skip[..., dh:dh+xh, dw:dw+xw]
        x = torch.cat([x, skip], dim=1)
        x = self.merge(x)
        x = self.res1(x, temb)
        if self.use_attention:
            x = x + self.attention(x)         
        x = self.res2(x, temb)
        return x
# give the output in 3 channels as the prediction
class OutHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.act  = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #zero-init for stable start
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(self.act(self.norm(x)))

class UNet(nn.Module):
    def __init__(self, in_ch=3, base=64, temb_dim=256, out_ch=3):
        super().__init__()
        self.time_mlp = time_embed(dim=temb_dim, out_dim=temb_dim)

        # Stem
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        # Encoder
        self.down0 = DownsampleBlock(base,     base*2, temb_dim, is_last=False)  # H -> H/2
        self.down1 = DownsampleBlock(base*2,   base*4, temb_dim, is_last=False)  # H/2 -> H/4
        self.down2 = DownsampleBlock(base*4,   base*8, temb_dim, is_last=False)  # H/4 -> H/8
        self.down3 = DownsampleBlock(base*8,   base*8, temb_dim, is_last=True)   # keep H/8

        # Bottleneck
        self.mid = BottleneckBlock(channels=base*8, temb_dim=temb_dim, use_attention=True)

        # Decoder
        self.up2 = UpsampleBlock(in_channels=base*8, skip_channels=base*8, out_channels=base*4, temb_dim=temb_dim, use_attention=True)
        self.up1 = UpsampleBlock(in_channels=base*4, skip_channels=base*4, out_channels=base*2, temb_dim=temb_dim, use_attention=True)
        self.up0 = UpsampleBlock(in_channels=base*2, skip_channels=base*2, out_channels=base,   temb_dim=temb_dim, use_attention=False)

        # Output head
        self.out = OutHead(base, out_ch)

    def forward(self, x, t):
        
        temb = self.time_mlp(t)  
        x = self.stem(x)
        x, s0 = self.down0(x, temb)   # x: H/2
        x, s1 = self.down1(x, temb)   # x: H/4
        x, s2 = self.down2(x, temb)
        x, _  = self.down3(x, temb)   # keep H/8; skip unused
        x = self.mid(x, temb)           # H/8
        x = self.up2(x, s2, temb)       # H/4
        x = self.up1(x, s1, temb)       # H/2
        x = self.up0(x, s0, temb)       # H
        out = self.out(x)

        return out
mse = nn.MSELoss()

@torch.no_grad()
def _rand_t(batch_size, T, device):
    return torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)

def diffusion_loss_vpred(model, x0, T, alphas, alphas_cumprod):
    N = x0.size(0)
    t = _rand_t(N, T, x0.device)
    x_t, eps = q_sample(x0, t, alphas_cumprod)  # returns x_t, ε

    a_bar = alphas_cumprod[t].view(-1,1,1,1)
    # v target
    v_target = a_bar.sqrt() * eps - (1.0 - a_bar).sqrt() * x0

    v_pred = model(x_t, t)
    return mse(v_pred, v_target)
def v_to_eps(v_pred, x_t, t, alphas, alphas_cumprod):
    
    a_bar = alphas_cumprod[t].view(-1,1,1,1)
    # For v-pred: eps_hat = sqrt(a_bar) * v_hat + sqrt(1 - a_bar) * x_t
    return a_bar.sqrt() * v_pred + (1.0 - a_bar).sqrt() * x_t
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}

# Total number of diffusion steps
T = 1000
# Schedules
betas, alphas, alphas_cumprod = make_betas_schedule_cosine(T, device=device)

# buffers
sqrt_alphas      = torch.sqrt(alphas)
sqrt_one_minus_c = torch.sqrt(1.0 - alphas_cumprod)
sqrt_recip_alph  = torch.sqrt(1.0 / alphas)
posterior_var    = betas 

# Build model
net = UNet(in_ch=3, base=64, temb_dim=256, out_ch=3).to(device)

# Optimizer & scheduler
optimizer = optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999))
scaler = amp.GradScaler(enabled=(device.type == "cuda"))
ema = EMA(net, decay=0.999)

epochs = 100
log_every = 100
global_step = 0
@torch.no_grad()
def ddpm_sample(model, shape, T, betas, alphas, alphas_cumprod, device):
    N, C, H, W = shape
    x_t = torch.randn(N, C, H, W, device=device)

    # Precompute a_bar_prev (alpha_bar at t-1) for posterior variance
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device, dtype=alphas_cumprod.dtype),
         alphas_cumprod[:-1]], dim=0
    )

    for t_int in reversed(range(T)):
        # integer timesteps (as in your current time_embed)
        t = torch.full((N,), t_int, device=device, dtype=torch.long)
        v_pred = model(x_t, t) 
        eps_pred = v_to_eps(v_pred, x_t, t, alphas, alphas_cumprod)
        a_t   = alphas[t_int]
        a_bar = alphas_cumprod[t_int]
        a_bar_prev = alphas_cumprod_prev[t_int]
        b_t   = betas[t_int]

        mean = (1.0 / torch.sqrt(a_t)) * (x_t - (b_t / torch.sqrt(1.0 - a_bar)) * eps_pred)

        if t_int > 0:
            posterior_var = b_t * (1.0 - a_bar_prev) / (1.0 - a_bar)
            x_t = mean + torch.sqrt(posterior_var) * torch.randn_like(x_t)
        else:
            x_t = mean

    return x_t
net.train()
for epoch in range(1, epochs+1):
    for i, (x0, _) in enumerate(dataloader):
        x0 = x0.to(device)  

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast('cuda', enabled=(device.type == "cuda")):
            loss = diffusion_loss_vpred(net, x0, T, alphas, alphas_cumprod)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        ema.update(net)

        if global_step % log_every == 0:
            print(f"epoch {epoch} step {global_step}  loss {loss.item():.4f}")
        global_step += 1
    if (epoch+1) % 1 == 0:
        # Save a checkpoint every epoch
        ckpt = {
            "net": net.state_dict(),
            "ema": ema.shadow,
            "opt": optimizer.state_dict(),
            "step": global_step,
            "epoch": epoch,
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
        }
        torch.save(ckpt, f"ddpm_epoch_{epoch}.pt")
        net.eval()
        ema.apply_to(net)  # use EMA weights for nicer samples

        samples = ddpm_sample(
            model=net,
            shape=(16, 3, 64, 64),  # batch of 16 images
            T=T,
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            device=device
        )

        ema.restore(net)   # restore training weights
        net.train()        # back to train mode

        # De-normalize to [0,1] and save grid
        grid = (samples.clamp(-1,1) + 1) * 0.5
        vutils.save_image(grid, f"samples_epoch_{epoch}.png", nrow=4)
        grid_np = (grid.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        plt.figure(figsize=(6,6))
        plt.axis("off")
        plt.imshow(grid_np[0])  # show just the first sample
        plt.show()
net.eval()
ema.apply_to(net)  

samples = ddpm_sample(
    model=net,
    shape=(16, 3, 64, 64),
    T=T,
    betas=betas,
    alphas=alphas,
    alphas_cumprod=alphas_cumprod,
    device=device
)

ema.restore(net)   # restore original (non-EMA) params

# De-normalize to [0,1] for viewing
grid = (samples.clamp(-1,1) + 1) * 0.5
vutils.save_image(grid, "samples_epoch_last.png", nrow=4)

img_path = "samples_epoch_last.png"
print(f"Saved grid to: {img_path}")

#display inline (Jupyter/Colab/etc.)
grid_np = np.array(Image.open(img_path))
plt.figure(figsize=(6,6))
plt.axis("off")
plt.imshow(grid_np)
plt.show()
