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
In contrast to the initial datasets (cifar, imagenet and stl10), the EuroSAT MSI dataset consists of 13 channels.
```
@article{helber2017eurosat,
   title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
   author={Helber, et al.},
   journal={ArXiv preprint arXiv:1709.00029},
   year={2017}
}
```

### Domain Adaption
