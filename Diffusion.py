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
celeba_root = "/kaggle/input/images"
dataset = datasets.ImageFolder(root=celeba_root, transform=transform)

#Dataloader
subset_size = 10000
batch = 16
all_indices = list(range(len(dataset)))
random_indices = random.sample(all_indices, subset_size)
subset = Subset(dataset, random_indices)
dataloader = DataLoader(subset, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)

imgs, _ = next(iter(dataloader))
print(imgs.shape == (batch, 3, 64, 64))

# Cumprod = cumulative product
# 1e-4, 2e-2 good for diffusion, start small, end with bigger varience

# beta_schedule to define variables for the math to come

def make_betas_schedule(T, device=None):
    betas = torch.linspace(1e-4, 2e-2, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

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
