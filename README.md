# DeepMuon

This repository is the Pytorch implementation of the article "DeepMuon: Accelerating Cosmic-Ray Muon Simulation Based on Optimal Transport". In this article, we propose a method for generating cosmic-ray muons using the principle of optimal transport. This repository includes the training and inference parts of the model and provides our sea-level muon generation weights. The training code is still being organized, and we will update more detailed training code in the future.

[English](./README.md) | [中文](./README_zh.md)

## Installation
```bash
git clone https://github.com/wangab0/deepmuon.git
cd deepmuon
conda create -n deepmuon python=3.9
conda activate deepmuon
pip install -r requirements.txt
```

## Model Inference
If you want to use our model to generate sea-level cosmic-ray muons, please first download our model parameters. The download link is as follows:

https://pan.baidu.com/s/16M1vqyeSlDLviU17OumLfw?pwd=18ri 

After downloading, store the model parameters in the `ckpts` directory, enter the `infer` directory, and run `muon_generator.py` to generate sea-level cosmic-ray muon events.

In subsequent versions, we will provide the GEANT4 integration code for DeepMuon.

## Model Training

Before training the model, you first need to prepare the training data. The data format is a 2D numpy array with dimensions Nx3, where N represents the number of muon events, and each event includes the energy and the cosine values of the angles between the velocity and the x-axis and y-axis.

The event energy needs to be transformed using the inverse Box-cox transformation to reduce skewness. Our energy transformation code is as follows:
```python
import scipy.special as special
transformed_energy = special.inv_boxcox(energy,9)
mean=torch.mean(transformed_energy)
transformed_energy=torch.tanh(transformed_energy-mean)
```
After preparing the training data, move the data to the `datas` directory to start training. Modify the contents of `train.sh` and run it to quickly begin training.