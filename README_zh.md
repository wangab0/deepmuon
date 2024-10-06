# DeepMuon

本仓库为文章DeepMuon: Accelerating Cosmic-Ray Muon Simulation Based on Optimal Transport的Pytorch实现,本文我们提出了一个利用最优传输原理实现宇宙线缪子产生的方法, 本仓库包含模型的训练与推理部分, 并提供了我们的海平面缪子生成权重。训练部分的代码正在持续整理中，后续我们将更新更详细的训练代码。

[English](./README.md) | [中文](./README.zh.md)

## 安装
```bash
git clone https://github.com/wangab0/deepmuon.git
cd deepmuon
conda create -n deepmuon python=3.9
conda activate deepmuon
pip install -r requirements.txt
```

## 模型推理
如果想使用我们的模型产生海平面宇宙线缪子，请首先下载我们的模型参数，下载链接如下：

https://pan.baidu.com/s/16M1vqyeSlDLviU17OumLfw?pwd=18ri 

下载完成后将模型参数储存到`ckpts`目录下，进入`infer`目录并运行`muon_generator.py`，即可产生海平面宇宙线缪子事例

后续版本中我们将提供DeepMuon的GEANT4集成代码

## 模型训练

模型训练前首先需要准备训练数据, 数据格式为一个维度为Nx3的二维numpy array, 其中N代表缪子事例数目, 每个事例包含能量以及速度与x轴和y轴方向的夹角余弦值 

事例能量需要使用逆Box-cox变换来降低风度，我们的能量变换代码如下：
```python
import scipy.special as special
transformed_energy = special.inv_boxcox(energy,9)
mean=torch.mean(transformed_energy)
transformed_energy=torch.tanh(transformed_energy-mean)
```
准备好训练数据后将数据移动到datas目录下即可开始训练，修改`train.sh`中的内容并运行即可快速开始
