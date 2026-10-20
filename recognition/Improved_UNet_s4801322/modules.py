import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

class U_Net(nn.Module):
    def __init__(self, latent_dimension):
        super().__init__()
        
        self.latent_dimension = latent_dimension