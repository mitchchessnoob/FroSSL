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
import wandb
import argparse
from WSOL.utils import dataloader, WSOL_model, attention, calculate_attention_metrics, safe_tensor_to_numpy

def main(dataset, model, key):
  train_iter, test_iter, classes = dataloader(dataset)
  model = WSOL_model(model, dataset)
  model = model.cuda()
  num_epochs = 5
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr=0.01,
      momentum=0.9,
      weight_decay=1e-4
  )
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,78,eta_min=0.001)
  best_acc = 0
  if wandb.api.api_key is None:
    wandb.login(key=key)
  try:
      wandb.init(project="Weakly-Supervised Object Localization", id="cifar10 resnet18 not pretrained",\
                 resume="auto", settings=wandb.Settings(init_timeout=300),\
                 config= {"learning_rate": 0.01,
                          "architecture": "resnet18 not pretrained",
                          "dataset": "CIFAR-10",
                          "epochs": num_epochs,
                          },
                 mode = "online"
  
                 )
  
      losses = []
      acces = []
      v_losses = []
      v_acces = []
  
      for epoch in tqdm(range(num_epochs)):
          epoch_loss = 0.0
          acc = 0.0
          var = 0.0
          model.train()
          train_pbar = train_iter
          for i, (x, _label) in enumerate(train_pbar):
              x = x.cuda(non_blocking=True)
              _label = _label.cuda(non_blocking=True)
              label = F.one_hot(_label, num_classes=10).float()
              seg_out = model(x)
  
              attn = attention(seg_out)
              # Smooth Max Aggregation
              logit = torch.log(torch.exp(seg_out*0.5).mean((-2,-1)))*2
              loss = criterion(logit, label)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              lr_scheduler.step()
              epoch_loss += loss.item()
              acc += (logit.argmax(-1)==_label).sum()
  
              wandb.log({
                'train_loss_step': loss,
                'train_accuracy_step': acc/(i*len(_label))
              })
              del x, _label, label, seg_out, attn, logit, loss
              #train_pbar.set_description('Accuracy: {:.3f}%'.format(100*(logit.argmax(-1)==_label).float().mean()))
  
          avg_loss = epoch_loss / (i + 1)
          losses.append(avg_loss)
          avg_acc = acc.cpu().detach().numpy() / (len(trainset))
          acces.append(avg_acc)
          wandb.log({
                'epoch': epoch,
                'train_loss_epoch': avg_loss,
                'train_accuracy_epoch': avg_acc
              })
          model.eval()
  
          epoch_metrics = {
              'loss': [],
              'accuracy': [],
              'mean_coverage': [],
              'peak_response': [],
              'smoothness': [],
              'peak_to_background': [],
              'attention_contiguity': [],
              'class_attention_std': [],
              'activation_consistency': []
          }
          test_pbar = tqdm(test_iter)
          for i, (x, _label) in enumerate(test_pbar):
              x = x.cuda(non_blocking=True)
              _label = _label.cuda(non_blocking=True)
              label = F.one_hot(_label, num_classes = 10).float()
              seg_out = model(x)
              attn = attention(seg_out)
              logit = torch.log(torch.exp(seg_out*0.5).mean((-2,-1)))*2
              loss = criterion(logit, label)
  
  
              test_pbar.set_description(
                  f'Test Acc: {100*acc:.2f}% | Loss: {loss.item():.4f}'
              )
  
              del  _label, label, logit, loss, acc
              torch.cuda.empty_cache()
  
  
          plt.close('all')
  
          epoch_averages = {
              key: np.mean(safe_tensor_to_numpy(values))
              for key, values in epoch_metrics.items()
          }
  
          # Log epoch metrics to wandb
          wandb.log({
              'epoch': epoch,
              'test/epoch_loss': epoch_averages['loss'],
              'test/epoch_accuracy': epoch_averages['accuracy'],
              'attention/epoch_mean_coverage': epoch_averages['mean_coverage'],
              'attention/epoch_peak_response': epoch_averages['peak_response'],
              'attention/epoch_smoothness': epoch_averages['smoothness'],
              'attention/epoch_peak_to_background': epoch_averages['peak_to_background'],
              'attention/epoch_contiguity': epoch_averages['attention_contiguity'],
              'attention/epoch_class_std': epoch_averages['class_attention_std'],
              'attention/epoch_activation_consistency': epoch_averages['activation_consistency']
          })
  
          if epoch_averages['accuracy'] > best_acc:
            best_acc = epoch_averages['accuracy']
            torch.save(model.state_dict(), './model_not_pretrained.pth')
  
          # conf = torch.max(nn.functional.softmax(seg_out, dim=1), dim=1)[0]
          # hue = (torch.argmax(seg_out, dim=1).float() + 0.5)/10
          # x -= x.min()
          # x /= x.max()
          # gs_im = x.mean(1)
          # gs_mean = gs_im.mean()
          # gs_min = gs_im.min()
          # gs_max = torch.max((gs_im-gs_min))
          # gs_im = (gs_im - gs_min)/gs_max
          # hsv_im = torch.stack((hue.float(), attn.squeeze().float(), gs_im.float()), -1)
          # im = hsv_to_rgb(hsv_im.cpu().detach().numpy())
          # ex = make_grid(torch.tensor(im).permute(0,3,1,2), normalize=True, nrow=25)
          # attns = make_grid(attn, normalize=False, nrow=25)
          # attns = attns.cpu().detach()
          # inputs = make_grid(x, normalize=True, nrow=25).cpu().detach()
          # display.clear_output(wait=True)
          # plt.figure(figsize=(20,8))
          # plt.imshow(np.concatenate((inputs.numpy().transpose(1,2,0),ex.numpy().transpose(1,2,0), attns.numpy().transpose(1,2,0)), axis=0))
  
          # plt.xticks(np.linspace(18,324,10), classes)
          # plt.xticks(fontsize=20)
          # plt.yticks([])
          # plt.title('CIFAR10 Epoch:{:02d}, Train:{:.3f}, Test:{:.3f}'.format(epoch, avg_acc, epoch_averages['accuracy']), fontsize=20)
          # display.display(plt.gcf())
  
          torch.cuda.empty_cache()
          gc.collect()
      wandb.finish()
  except Exception as e:
      wandb.finish()
      raise e
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run semi-supervised training.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--key", type=str, required=True, help="WandB API key.")
    model = resnet18(pretrained=False)
    args = parser.parse_args()
    main(args.dataset, model, args.key)
