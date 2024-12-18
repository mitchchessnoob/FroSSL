import sys
import argparse
import os
import torch.nn as nn
import sys
from torchvision import models
import torch
from solo.args.linear import parse_cfg
from solo.data.pretrain_dataloader import (
    NCropAugmentation,
    build_transform_pipeline,
    prepare_datasets,
    prepare_dataloader,
    prepare_n_crop_transform
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from semi_supervised.utils.evaluate_std import evaluate
from semi_supervised.utils.dataset_download import dataset
from semi_supervised.utils.model import SSLModel
from semi_supervised.utils.train_step_emstd import train_step
from semi_supervised.utils.optim_sch import create_optimizer_and_scheduler
from semi_supervised.utils.unlabeled_dataset import flatten_image_directory
from solo.data.classification_dataloader import prepare_transforms
from solo.utils.lars import LARS
from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.losses.frossl import multiview_frossl_loss_func
import yaml
from omegaconf import OmegaConf
import wandb
import numpy as np

def main(configs_path):
    try:
        cfg = OmegaConf.load(configs_path)
        configs = parse_cfg(cfg)

        dataset()
        
        
        # Initialize wandb
        wandb.init(project=configs.name, name=f"SemiSL without FroSSL and standard training-pretrained-D/A")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform_train = transforms.Compose([
                
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            
                
                transforms.RandomHorizontalFlip(p=0.5),

        
                transforms.RandomAffine(
                    degrees=10,        
                    translate=(0.1, 0.1),  
                    scale=(0.9, 1.1),  
                    shear=10           
                ),
            
                
                transforms.RandomResizedCrop(
                    size=(224, 224),  
                    scale=(0.8, 1.0),  
                    ratio=(0.9, 1.1)   
                ),
            
                
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            
                
                transforms.ToTensor(),
            
                
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = ImageFolder(root=configs.data.labeled_path, transform=transform_train)

       
        test_dataset = ImageFolder(root=configs.data.test_path, transform=transform_test)
        
        # Create DataLoader for training
        labeled_loader = DataLoader(train_dataset, batch_size=configs.optimizer.batch_size, shuffle=True, num_workers=configs.data.num_workers)
        
        # Create DataLoader for testing
        test_loader = DataLoader(test_dataset, batch_size=configs.optimizer.batch_size, shuffle=False, num_workers=configs.data.num_workers)
        
        # Create model
        num_classes = len(train_dataset.classes)
        model = models.resnet18(pretrained=True).to(device)

        in_features = model.fc.in_features
        
        model.fc = nn.Identity()  # Remove the original FC layer
        
        # Add a new FC layer for classification
        model.classifier = nn.Linear(in_features, num_classes)
        
        model = model.to(device)
        # Define optimizer and criterion
        optimizer, scheduler = create_optimizer_and_scheduler(model, configs)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        num_epochs = configs.max_epochs
        best_acc = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = train_step(model, labeled_loader,
                                  optimizer, scheduler, criterion, device, epoch, configs)

            test_loss, test_acc = evaluate(model, test_loader, criterion, device, epoch)

            print(f'Epoch: {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            os.makedirs(configs.checkpoint.dir, exist_ok=True)
            if test_acc > best_acc and configs.checkpoint.enabled:
                best_acc = test_acc
                torch.save(model.state_dict(), f"{configs.checkpoint.dir}/best_model.pth")

        # Save final model
        torch.save(model.state_dict(), f"{configs.checkpoint.dir}/final_model.pth")

        wandb.finish()
    except Exception as e:
        wandb.finish()
        raise e
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run semi-supervised training.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file.")
    
    args = parser.parse_args()
    main(args.config_path)
