from torchvision.models import resnet18
from omegaconf import OmegaConf
import torch
import sys
import torch.nn as nn
from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.checkpointer import Checkpointer
from solo.utils.checkpointer import Checkpointer
from solo.args.linear import parse_cfg

def get_model(type, dataset, ckpt_path = None):
  if type == "pretrained":
    return resnet18(pretrained=True)
  elif type == "not pretrained":
    return resnet18(pretrained=False)
  elif type == "FroSSL":
      if dataset == "cifar10":
        cfg = "/content/FroSSL/scripts/pretrain/cifar/frossl.yaml"
      elif dataset == "STL10":
        cfg = "/content/FroSSL/scripts/pretrain/stl10/frossl.yaml"
    
      cfg = OmegaConf.load(cfg)

      cfg = parse_cfg(cfg)
      
      backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
      
      # initialize backbone
      model = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
      if cfg.backbone.name.startswith("resnet"):
              # remove fc layer
              model.fc = nn.Identity()
              cifar = cfg.data.dataset in ["cifar10", "cifar100"]
              if cifar:
                  model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
                  model.maxpool = nn.Identity()
      
      assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
      
      state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
      for k in list(state.keys()):
          if "encoder" in k:
              state[k.replace("encoder", "backbone")] = state[k]
          if "backbone" in k:
              state[k.replace("backbone.", "")] = state[k]
          del state[k]
      model.load_state_dict(state, strict=False)
      return model
  exit("model not allowed")
