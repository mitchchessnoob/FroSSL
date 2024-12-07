# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder, EuroSAT
from datasets import load_dataset

try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True


class EuroSATDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def build_custom_pipeline():
    """Builds augmentation pipelines for custom data.
    If you want to do exoteric augmentations, you can just re-write this function.
    Needs to return a dict with the same structure.
    """

    pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }
    return pipeline


def prepare_transforms(dataset: str) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    tiny_imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=64, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    eurosat_rgb_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=64, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(), # [0, 255] -> [0, 1] !!
                transforms.Normalize((0.3127, 0.3451, 0.3703), (0.1914, 0.1270, 0.1067)), # (data["image"].float()/255).mean(dim=(0, 2, 3))
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.3127, 0.3451, 0.3703), (0.1914, 0.1270, 0.1067)),
            ]
        ),
    }



    eurosat_msi_pipeline = {
        "T_train": transforms.Compose(
            [   transforms.Lambda(lambda x: x.permute(2, 0, 1)),
                transforms.Lambda(lambda x: x/10_000),
                transforms.RandomResizedCrop(size=64, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize((0.1354, 0.1118, 0.1043, 0.0948, 0.1199, 0.2000, 0.2369, 0.2297, 0.0732, 0.0012, 0.1819, 0.1119, 0.2594), # (data["image"].float()/10_000).mean(dim=(0, 2, 3))
                                     (0.0246, 0.0333, 0.0395, 0.0594, 0.0566, 0.0861, 0.1087, 0.1118, 0.0405, 0.0005, 0.1003, 0.0761, 0.1232)), 
            ]
        ),
        "T_val": transforms.Compose(
            [   transforms.Lambda(lambda x: x.permute(2, 0, 1)),
                transforms.Lambda(lambda x: x/10_000),
                transforms.Normalize((0.1354, 0.1118, 0.1043, 0.0948, 0.1199, 0.2000, 0.2369, 0.2297, 0.0732, 0.0012, 0.1819, 0.1119, 0.2594),
                                     (0.0246, 0.0333, 0.0395, 0.0594, 0.0566, 0.0861, 0.1087, 0.1118, 0.0405, 0.0005, 0.1003, 0.0761, 0.1232)),
            ]
        ),
    }

    custom_pipeline = build_custom_pipeline()

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "tiny-imagenet": tiny_imagenet_pipeline,
        "custom": custom_pipeline,
        "eurosat_rgb": eurosat_rgb_pipeline,
        "eurosat_msi": eurosat_msi_pipeline
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val


def apply_transformation(example, transformation): # for huggingface dataset
    example["image"] = transformation(example["image"])
    return example



def convert_image_dataset_to_embedding_dataset(dataloader,ssl_model, desc):
    embeddings = []
    labels = []
    shuffle = "val" not in desc

    with torch.no_grad():
        ssl_model.eval()
        ssl_model.cuda()


        for batch in tqdm.tqdm(dataloader, desc=f"Converting {desc} set to embeddings"):
            x, y = batch
            x = x.cuda(non_blocking=True)
        
            embedding = ssl_model.backbone(x)

            embeddings.extend(embedding.cpu().detach().squeeze().numpy())
            labels.extend(y.cpu().detach().squeeze().numpy())

    
    embeddings = torch.tensor(np.array(embeddings))
    labels = torch.tensor(np.array(labels), dtype=torch.int64)
    tensor_dataset = torch.utils.data.TensorDataset(embeddings, labels)

    return torch.utils.data.DataLoader(tensor_dataset, 
                                       batch_size=dataloader.batch_size, 
                                       shuffle=shuffle, 
                                       num_workers=dataloader.num_workers, 
                                       pin_memory=dataloader.pin_memory, 
                                       drop_last=dataloader.drop_last)

def prepare_test_data(
    dataset: str,
    train_data_path: Optional[Union[str, Path]] = None,
    val_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
    data_fraction: float = -1.0,
    auto_augment: bool = False,
    train_pipeline: Optional[Callable] = None,
    val_pipeline: Optional[Callable] = None,
    precompute_features: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
        auto_augment (bool, optional): use auto augment following timm.data.create_transform.
            Defaults to False.
        train_pipeline (Optional[Callable], optional): pipeline of transformations for training dataset.
            Defaults to None.
        val_pipeline (Optional[Callable], optional): pipeline of transformations for validation dataset.
            Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader.
    """

    #_, T_test = prepare_transforms(dataset)

    # if auto_augment:
    #     T_train = create_transform(
    #         input_size=224,
    #         is_training=True,
    #         color_jitter=None,  # don't use color jitter when doing random aug
    #         auto_augment="rand-m9-mstd0.5-inc1",  # auto augment string
    #         interpolation="bicubic",
    #         re_prob=0.25,  # random erase probability
    #         re_mode="pixel",
    #         re_count=1,
    #         mean=IMAGENET_DEFAULT_MEAN,
    #         std=IMAGENET_DEFAULT_STD,
    #     )
    if dataset == "eurosat_msi":
        T_test = transforms.Compose(
            [   transforms.Lambda(lambda x: x.permute(2, 0, 1)),
                transforms.Lambda(lambda x: x/10_000),
                transforms.Normalize((0.1354, 0.1118, 0.1043, 0.0948, 0.1199, 0.2000, 0.2369, 0.2297, 0.0732, 0.0012, 0.1819, 0.1119, 0.2594),
                                     (0.0246, 0.0333, 0.0395, 0.0594, 0.0566, 0.0861, 0.1087, 0.1118, 0.0405, 0.0005, 0.1003, 0.0761, 0.1232)),
            ])
        test_dataset = load_dataset("blanchon/EuroSAT_MSI", split="test")
        test_dataset.set_format("torch", columns=["image", "label"])
        # transform into a torch Dataset
        test_dataset = EuroSATDataset(test_dataset, transform=T_test)

        test_loader = DataLoader(
            test_dataset,
            batch_size=5400,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
    elif dataset=="mit67":
        T_test= transforms.Compose(
            [
                transforms.Resize(224),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4887, 0.4314, 0.3724), std=(0.2378, 0.2332, 0.2292)),
            ])
        test_dataset = torchvision.datasets.ImageFolder(root=val_data_path, transform=T_test)
        test_loader = DataLoader(
                test_dataset,
                batch_size=len(test_dataset),
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )


    # if precompute_features:
    #     assert model is not None

    #     train_loader = convert_image_dataset_to_embedding_dataset(train_loader, model, desc="precomputing train dataloader")
    #     val_loader = convert_image_dataset_to_embedding_dataset(val_loader, model, desc="precomputing val dataloader")

    return test_loader
