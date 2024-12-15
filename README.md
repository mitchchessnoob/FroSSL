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
 In those files the hyperparameters are defined. The pretrain conifguration further refers to a augmentation configuration.
- finally, `sh run_experiment.sh`
