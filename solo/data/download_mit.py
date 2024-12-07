import shutil
import os
from torchvision import datasets
import torch
import kagglehub


def download_mit_dataset():

    path = kagglehub.dataset_download("itsahmad/indoor-scenes-cvpr-2019")
    current_dir = os.getcwd()
    shutil.move(path, current_dir+"\\datasets\\mit67")

    with open(current_dir+"\\datasets\\mit67\\TrainImages.txt", 'r') as f:
        train_files = f.readlines()
    train_files = [fname[:-1].split("/") for fname in train_files[:-1]] +[train_files[-1].split("/")]

    with open(current_dir+"\\datasets\\mit67\\TestImages.txt", 'r') as f:
        test_files = f.readlines()
    test_files = [fname[:-1].split("/") for fname in test_files[:-1]] +[test_files[-1].split("/")]

    move_images(current_dir+"\\datasets\\mit67\\indoorCVPR_09\\Images", train_files, current_dir+"\\datasets\\mit67\\train_images")
    move_images(current_dir+"\\datasets\\mit67\\indoorCVPR_09\\Images", test_files, current_dir+"\\datasets\\mit67\\test_images")


def move_images(images_dir, file_list, target_dir):
    for label, file_name in file_list:
        src_path = os.path.join(images_dir, label, file_name)
        dest_path = os.path.join(target_dir, label, file_name)
        os.makedirs(os.path.join(target_dir, label), exist_ok=True)
        try: # the given train/test split does not use all images!!!
            shutil.move(src_path, dest_path)
        except:
            pass
