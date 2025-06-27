The full technical report for this project can be accessed here:

[📄 View Full Report (PDF)](FULL%20REPORT/DL_advanced_Report.pdf)

# Experiments with FroSSL (Frobenius Norm Minimization for Efficient Multiview Self-Supervised Learning)

This is Project is build up on a fork of the official PyTorch [implementation](https://github.com/OFSkean/FroSSL) of the [FroSSL paper](https://arxiv.org/pdf/2310.02903), which started as a fork of the fantastic [solo-learn](https://github.com/vturrisi/solo-learn.git) library.
We adapt the official implementation for different tasks, see below.


```
@inproceedings{skean2024frossl,
  title={FroSSL: Frobenius Norm Minimization for Self-Supervised Learning},
  author={Skean, Oscar and Dhakal, Aayush and Jacobs, Nathan and Giraldo, Luis Gonzalo Sanchez},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

## Tasks

### Classification of Satelite footage
The authors of the FroSSL paper applied their method to object oriented classification with datasets like cifar, imagenet and stl10. In our experiments we tested the method in scene classificaiton with the EuroSAT dataset, containing 64x64x13 satelite images with 10 different classes.
```
@article{helber2017eurosat,
   title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
   author={Helber, et al.},
   journal={ArXiv preprint arXiv:1709.00029},
   year={2017}
}
```

### Domain Adaption
For this task we trained in a Semi-Supervised Learning framework a model that handles labeled and unlabeled data coming from different domains. The unlabeled loss is represented by FroSSL objective. We then evaluate our model on samples coming from an unseen distribution during trainig. Our work has been inspired by
```
@misc{liang2024generalizedsemisupervisedlearningselfsupervised,
      title={Generalized Semi-Supervised Learning via Self-Supervised Feature Adaptation}, 
      author={Jiachen Liang and Ruibing Hou and Hong Chang and Bingpeng Ma and Shiguang Shan and Xilin Chen},
      year={2024},
      eprint={2405.20596},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.20596}, 
}
```
## how to run an experiment
### setup
```
git clone -b main https://github.com/mitchchessnoob/FroSSL
cd FroSSL
pip install -r requirements.txt
wandb login 
```
Further if you wan to use the MIT67-dataset:
`sh install_mit67.sh`
### scene classification
- configure experiment by adapting the run_experiment.sh file. the CONFIG_NAME variable refers to two yaml files
  - scripts/pretrain/DATASET/CONFIG_NAME.yaml
  - scripts/linear/DATASET/CONFIG_NAME.yaml. \
 In those files the hyperparameters are defined. The pretrain conifguration further refers to an augmentation configuration.
- finally, `sh run_experiment.sh`

### domain adaption

- Configure experiment by adapting the main_SL.sh file:
  - CONFIGS_PATH variable refers to a yaml file at semi_supervised/configs.yaml that contains all settings regarding the method, dataset path, hyperparameters, ...
  - AUGMENTS_PATH variable refers to a yaml file at semi_supervised/asymmetric.yaml that contains all augmentation parameters for labeled and unlabeled sets
  - INSERT you kaggle credentials if you want to download datasets from kaggle.
  - It's assumed that the three datasets are in the directories specified in CONFIGS_PATH, the code automatically downloads office31 dataset, change the function **dataset()** in semi_supervised/utils/dataset_download.py for training on a different dataset.
  - It is assumed that labeled and test sets are image folders, while unlabeled dataset has to be a flat directory (if it's an image folder the code automatically transforms it in the required shape)
  - In semi_supervised/utils/model.py you can change the backbone or set it to pretrained = False
  - finally, `sh main_SL.sh` \

- If you want to train a supervised model, use main_SL_std.sh
   - It works the same way as the standard experiments
   - The only difference is that the augmentation pipeline is inside the main function main_SL_morestandard.py
   - finally, `sh main_SL_std.sh`
