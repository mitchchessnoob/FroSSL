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
from torchvision import datasets
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
from sklearn.model_selection import train_test_split

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


def make_train_val_split(dataset):
        targets = dataset.targets
        # Get the class indices for each class
        class_indices = {}
        for idx, target in enumerate(targets):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(idx)

        # Initialize lists to store the split indices
        train_indices = []
        val_indices = []

        # Split each class's indices into train and validation sets (e.g., 80/20 split)
        for class_label, indices in class_indices.items():
            train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)

        # Create train and validation subsets using the indices
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        return train_dataset, val_dataset


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
    office_home_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    ),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    ),
            ]
        ),
    }
    office31_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                    ),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        ),
    }
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


    mit67_pipeline = { 
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4887, 0.4314, 0.3724), std=(0.2378, 0.2332, 0.2292)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(224),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4887, 0.4314, 0.3724), std=(0.2378, 0.2332, 0.2292)),
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
        "office31": office31_pipeline,
        "office_home":office_home_pipeline,
        "eurosat_rgb": eurosat_rgb_pipeline,
        "eurosat_msi": eurosat_msi_pipeline,
        "mit67": mit67_pipeline
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val


def apply_transformation(example, transformation): # for huggingface dataset
    example["image"] = transformation(example["image"])
    return example


def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    train_data_path: Optional[Union[str, Path]] = None,
    val_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    download: bool = True,
    data_fraction: float = -1.0,
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if val_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        val_data_path = sandbox_folder / "datasets"

    assert dataset in ["office_home", "office31", "cifar10", "cifar100", "stl10", "imagenet", "imagenet100", "tiny-imagenet", "custom", "eurosat_rgb", "eurosat_msi", "mit67"]

    if dataset in ["office31", "office_home"]:
        train_dataset = ImageFolder(train_data_path, transform=T_train)
        val_dataset = ImageFolder(val_data_path, transform=T_val)
        
    elif dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            train_data_path,
            train=True,
            download=download,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            val_data_path,
            train=False,
            download=download,
            transform=T_val,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            train_data_path,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            val_data_path,
            split="test",
            download=download,
            transform=T_val,
        )

    elif dataset[:7] == "eurosat":
        if dataset[7:] == "_rgb":
            train_dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")
            val_dataset = load_dataset("blanchon/EuroSAT_RGB", split="validation")
        elif dataset[7:] == "_msi":
            train_dataset = load_dataset("blanchon/EuroSAT_MSI", split="train")
            val_dataset = load_dataset("blanchon/EuroSAT_MSI", split="validation")
            train_dataset.set_format("torch", columns=["image", "label"])
            val_dataset.set_format("torch", columns=["image", "label"])
        else:
            pass
        # transform into a torch Dataset
        train_dataset = EuroSATDataset(train_dataset, transform=T_train)
        val_dataset = EuroSATDataset(val_dataset, transform=T_val)

    elif dataset=="mit67":
        # ImageFolder lädt das Dataset automatisch basierend auf der Ordnerstruktur
        t_dataset = datasets.ImageFolder(root=train_data_path, transform=T_train)
        train_dataset, val_dataset = make_train_val_split(t_dataset)

    
    elif dataset in ["imagenet", "imagenet100", "custom", "tiny-imagenet"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = H5Dataset(dataset, train_data_path, T_train)
            val_dataset = H5Dataset(dataset, val_data_path, T_val)
        else:
            train_dataset = ImageFolder(train_data_path, T_train)
            val_dataset = ImageFolder(val_data_path, T_val)

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        data = train_dataset.samples
        files = [f for f, _ in data]
        labels = [l for _, l in data]

        from sklearn.model_selection import train_test_split

        files, _, labels, _ = train_test_split(
            files, labels, train_size=data_fraction, stratify=labels, random_state=42
        ) 
        train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset, val_dataset


def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader

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

def prepare_data(
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

    T_train, T_val = prepare_transforms(dataset)
    if train_pipeline is not None:
        T_train = train_pipeline
    if val_pipeline is not None:
        T_val = val_pipeline

    if auto_augment:
        T_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=None,  # don't use color jitter when doing random aug
            auto_augment="rand-m9-mstd0.5-inc1",  # auto augment string
            interpolation="bicubic",
            re_prob=0.25,  # random erase probability
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        data_format=data_format,
        download=download,
        data_fraction=data_fraction,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if precompute_features:
        assert model is not None

        train_loader = convert_image_dataset_to_embedding_dataset(train_loader, model, desc="precomputing train dataloader")
        val_loader = convert_image_dataset_to_embedding_dataset(val_loader, model, desc="precomputing val dataloader")

    return train_loader, val_loader
