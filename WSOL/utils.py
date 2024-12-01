from torchvision.datasets import ImageFolder, CIFAR10, STL10
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from tqdm import tqdm_notebook as tqdm
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import numpy as np
from IPython import display
import requests
from io import BytesIO
from PIL import Image
from PIL import Image, ImageSequence
from IPython.display import HTML
import warnings
from matplotlib import rc
import gc

def dataloader(str: dataset):
  if dataset == "cifar10":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
  
    trainset = CIFAR10(root='.', train=True, download=True, transform=transform_train)
    train_iter = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
  
    testset = CIFAR10(root='.', train=False, download=True, transform=transform_test)
    test_iter = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
  
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  elif dataset == "stl10":
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=24),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239)),
    ])
  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239)),
    ])
  
    trainset = STL10(root='.', split='train', download=True, transform=transform_train)
    train_iter = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
  
    testset = STL10(root='.', split='test', download=True, transform=transform_test)
    test_iter = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
  
    classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
  return train_iter, test_iter, classes


def create_frozen_resnet(model, num_classes=10, str: dataset):
    # Load pretrained model
    image_dims = {"cifar10":32, "stl10":96}
    image_dim = image_dims[dataset]
    # Replace first conv layer (this will be trainable)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

    # Get all layers except last two
    resnet_ = list(model.children())[:-2]

    # # Freeze all ResNet parameters except the first conv layer
    # for i, layer in enumerate(resnet_):
    #     if i != 0:  # Skip freezing the first layer (our new conv1)
    #         for param in layer.parameters():
    #             param.requires_grad = False

    # Replace the specified layer with upsampling (will be trainable)
    resnet_[3] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    # Create and initialize the classifier (will be trainable)
    classifier = nn.Conv2d(512, num_classes, 1)
    torch.nn.init.kaiming_normal_(classifier.weight)

    # Add final layers
    resnet_.append(classifier)
    resnet_.append(nn.Upsample(size=image_dim, mode='bilinear', align_corners=False))

    # Create the final model
    tiny_resnet = nn.Sequential(*resnet_)

    return tiny_resnet
  
def attention(x):
    return torch.sigmoid(torch.logsumexp(x,1, keepdim=True))
  
def calculate_attention_metrics(attn, seg_out, _label):
    # Basic attention metrics
    attention_stats = {
        'mean_coverage': attn.mean().item(),
        'peak_response': attn.max().item(),
        'smoothness': torch.mean(torch.abs(attn[:, :, 1:] - attn[:, :, :-1])).item(),

        # Add peak-to-background ratio
        'peak_to_background': (attn.max() / (attn.min() + 1e-6)).item(),

        # Add attention contiguity (using 2D gradients)
        'attention_contiguity': -torch.norm(torch.gradient(attn, dim=(2,3))[0]).mean().item(),

        # Add class-wise attention statistics
        'class_attention_std': torch.std(attn.view(attn.size(0), -1), dim=1).mean().item(),

        # Add activation consistency per class
        'activation_consistency': torch.std(seg_out, dim=(2,3)).mean().item()
    }
    return attention_stats

def safe_tensor_to_numpy(values):
    if isinstance(values, (list, tuple)):
        # Handle lists/tuples of tensors
        return [safe_tensor_to_numpy(v) for v in values]
    elif isinstance(values, torch.Tensor):
        # Move tensor to CPU if it's on CUDA
        if values.is_cuda:
            values = values.cpu()
        return values.detach().numpy()
    return values
