# Probabilistic-MCTrans
Code for Probabilistic Multi-Compound Transformer (Probabilistic MCTrans).

This code are modified and combine code from [Probabilistic-Unet-Pytorch](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch) and [MCTrans](https://github.com/JiYuanFeng/MCTrans). Basically this code add a probabilistic model from Probabilistic U-Net into transformer-based model MCTrans. In previous study, model MCTrans can solved long-range dependencies problem and model Probabilistic U-Net can capture ambiguity in biomedical image. For installation, you can check [MCTrans](https://github.com/JiYuanFeng/MCTrans) repository. 

## Probabilistic U-Net
Title: A Probabilistic U-Net for Segmentation of Ambiguous Images.<br>
Authors: Simon A. A. Kohl, Bernardino Romera-Paredes, Clemens Meyer, Jeffrey De Fauw, Joseph R. Ledsam, Klaus H. Maier-Hein, S. M. Ali Eslami, Danilo Jimenez Rezende, Olaf Ronneberger.<br>
[[paper](https://arxiv.org/abs/1806.05034)]<br>
The architecture of Probabilistic U-Net:<br>
Training process:<br>
![Training](https://github.com/rizalmaulanaa/Probabilistic-MCTrans/blob/master/model-img/Models-Probabilistic-U-Net-training.png?raw=true)<br>
Sampling process:<br>
![Sampling](https://github.com/rizalmaulanaa/AProbabilistic-MCTrans/blob/master/model-img/Models-Probabilistic-U-sNet-sampling.png?raw=true)<br>

## Multi-Compound Transformer (MCTrans)
Title: Multi-Compound Transformer for Accurate Biomedical Image Segmentation<br>
Authors: Yuanfeng Ji, Ruimao Zhang, Huijie Wang, Zhen Li, Lingyun Wu, Shaoting Zhang, Ping Luo.<br>
[[paper](https://arxiv.org/abs/2106.14385)]<br>
The architecture of MCTranst:<br>
Training process:<br>
![Training & Sampling](https://github.com/rizalmaulanaa/Probabilistic-MCTrans/blob/master/model-img/Models-MCTrans.png?raw=true)<br>

## Probabilistic Multi-Compound Transformer (Probabilistic MCTrans)
The architecture of Probabilistic MCTrans:<br>
Training process:<br>
![Training](https://github.com/rizalmaulanaa/Probabilistic-MCTrans/blob/master/model-img/Models-prob-MCTrans-training.png?raw=true)<br>
Sampling process:<br>
![Sampling](https://github.com/rizalmaulanaa/Probabilistic-MCTrans/blob/master/model-img/Models-prob-MCTrans-sampling.png?raw=true)<br>
