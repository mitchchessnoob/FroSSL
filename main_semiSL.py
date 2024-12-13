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
from semi_supervised.utils.evaluate import evaluate
from semi_supervised.utils.dataset_download import dataset
from semi_supervised.utils.model import SSLModel
from semi_supervised.utils.train_step import train_step
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

def main(configs_path, augments_path):
    try:
        cfg = OmegaConf.load(configs_path)
        configs = parse_cfg(cfg)

        with open(augments_path, "r") as f:
          yaml_data = f.read()
        dataset()
        # Augmentation pipeline
        augments = OmegaConf.create(yaml_data)
        
        # Initialize wandb
        wandb.init(project=configs.name, name=f"SemiSL FroSSL W/A NOT pretrained")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        # Get the transforms and num_crops from the augmentation file
        transforms_list = []
        num_crops_list = []

        for config in augments:
            transform = build_transform_pipeline(configs.data.dataset, config)  
            transforms_list.append(transform)
            num_crops_list.append(config['num_crops'])

        # Create the N-crop transform using function in solo library
        transform = prepare_n_crop_transform(transforms_list, num_crops_list)
        T_train, T_val = prepare_transforms(configs.data.dataset)

        # Prepare datasets
        labeled_dataset = prepare_datasets(
            dataset= configs.data.dataset,
            transform=transform,
            train_data_path=configs.data.labeled_path,
            data_format='image_folder'
        )
        #solo dataloader without labels wants a flat directory, but the downloaded datasets are not in this format
        # we therefore process it 
        proper_format_unlabeled = True
        for item in os.listdir(configs.data.unlabeled_path):
            if os.path.isdir(os.path.join(configs.data.unlabeled_path, item)):
                proper_format_unlabeled = False
        if proper_format_unlabeled:
            unlabeled_path = configs.data.unlabeled_path
        else:
            unlabeled_path = "./temporary_dir"
            flatten_image_directory(configs.data.unlabeled_path, unlabeled_path)
            
        unlabeled_dataset = prepare_datasets(
            dataset=configs.data.dataset,
            transform=transform,
            train_data_path=unlabeled_path,
            data_format='image_folder',
            no_labels=True
        )

        test_dataset = prepare_datasets(
            dataset=configs.data.dataset,
            transform=T_val,
            train_data_path=configs.data.test_path,
            data_format='image_folder',
            train_dataset=False
        )

        # Create the dataloaders
        labeled_loader = prepare_dataloader(labeled_dataset, num_workers = configs.data.num_workers,\
                                            batch_size=configs.optimizer.batch_size)
        unlabeled_loader = prepare_dataloader(unlabeled_dataset, num_workers = configs.data.num_workers,\
                                              batch_size=configs.optimizer.batch_size)
        test_loader = prepare_dataloader(test_dataset, num_workers = configs.data.num_workers,\
                                        batch_size=configs.optimizer.batch_size, shuffle=False)

        # Create model
        num_classes = len(labeled_dataset.classes)
        model = SSLModel(num_classes).to(device)

        # Optimizer and criterion
        optimizer, scheduler = create_optimizer_and_scheduler(model, configs)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        num_epochs = configs.max_epochs
        best_acc = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = train_step(model, labeled_loader, unlabeled_loader,
                                  optimizer, scheduler, criterion, device, epoch, configs)

            test_loss, test_acc = evaluate(model, test_loader, criterion, device, epoch)

            print(f'Epoch: {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            os.makedirs(configs.checkpoint.dir, exist_ok=True)
            if test_acc > best_acc and configs.checkpoint.enabled:
                best_acc = test_acc
                torch.save(model.state_dict(), f"{configs.checkpoint.dir}/best_model_WA_NOT.pth")

        # Save final model
        torch.save(model.state_dict(), f"{configs.checkpoint.dir}/final_model.pth")

        wandb.finish()
    except Exception as e:
        wandb.finish()
        raise e
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run semi-supervised training.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--augments_path", type=str, required=True, help="Path to the augmentations file.")
    
    args = parser.parse_args()
    main(args.config_path, args.augments_path)
