# Training Your Sparse Neural Network Better with Any Mask (ICML 2022)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
Pytorch Implementation of ICML 2022 

## Installation
We recommend users to use `conda` to install the running environment. The following dependencies are required:
```
CUDA=11.1
Python=3.7.7
pytorch=1.9.0
sklearn=1.0.1
pillow=8.3.1
opencv-python
svgpathtools
cycler==0.10.0
kiwisolver==1.1.0
matplotlib==3.1.1
protobuf==3.9.2
pyparsing==2.4.2
python-dateutil==2.8.0
pytz==2019.2
scipy==1.3.1
seaborn==0.9.0
six==1.12.0
tensorboardX==1.8
tqdm==4.36.1
```
Our code should be compatible with pytorch>=1.5.0


## How to create the sparse mask for various SOTA pruning methods ? 
### Using pruning_techniques directory included with this repository :
### Following is an example of creating a Lottery ticket mask :
```
python3 main.py --prune_type=lt --arch_type=resnet18 --dataset=cifar10 --prune_percent=10 --prune_iterations=5
```
- `--prune_type` : Type of pruning  
- `--arch_type`	 : Type of architecture
- `--dataset`	: Choice of dataset 
- `--prune_percent`	: Percentage of weight to be pruned after each cycle. 
- `--prune_iterations`	: Number of cycle of pruning that should be done. 
- `--lr`	: Learning rate 
- `--batch_size`	: Batch size 
- `--end_iter`	: Number of Epochs 
- `--gpu`	: Decide Which GPU the program should use 

## Pruning methods:

**RP:** random pruning

**OMP:** oneshot pruning, magnitude pruning

**GMP**: To prune, or not to prune: exploring the efficacy of pruning for model compression

**TP:** Detecting Dead Weights and Units in Neural Networks, Page19 Table2.1 Taylor1Scorer (adding abs in our implementation) 

**SNIP:** SNIP: Single-shot network pruning based on connection sensitivity

**GraSP:** Picking winning tickets before training by preserving gradient flow

**SynFlow:** Pruning neural networks without any data by iteratively conserving synaptic flow

## Code Details: 

pruning methods implemented in **pruning_utils.py**

**example.py** provides an simple examples

## Training using soft-activation
### Keep the mask identified using the previous pruning methods in the soft_activation/mask directory
```
python -u soft_activation/train_ticket.py --dataset cifar100 --activation swish  --arch resnet18  --manualSeed 42 --depth 18  --model [initial model path] --resume [resume_path] --save_dir [output_directory]  --gpu 3
```
### Activation based analysis
```
python activation_analysis.py --arch resnet18 --dataset cifar100 --manualSeed 42 --depth 18 --pretrained [pretrained checkpoint path]  --eval --gpu_id 1  --activation [relu/swish/mish] --layer [layer_number_to_analyse]
```
## Training using skip-connections
### Keep the mask identified using the previous pruning methods in the skip_connection/mask directory
```
python skip_connection/train_ticket.py --dataset cifar100 --activation [activation_to_use]  --arch resnet18  --manualSeed 42 --depth 18  --model [initial model path] --resume [resume_path] --save_dir [output_directory]  --gpu 3  --gpu 0
```

## Training using label-smoothening
### Keep the mask identified using the previous pruning methods in the label-smoothening/mask directory
```
python label-smoothening/train_ticket.py --dataset cifar100 --activation [activation_to_use]  --arch resnet18  --manualSeed 42 --depth 18  --model [initial model path] --resume [resume_path] --save_dir [output_directory]  --gpu 3  --gpu 0
```

## Training using LRsI
### Keep the mask identified using the previous pruning methods in the LRsI/mask directory
```
python train_ticket.py --dataset cifar100 --activation relu  --arch resnet18  --manualSeed 42 --depth 18  --model [initial model path] --resume [resume_path] --save_dir [output_directory]  --gpu 2 --gradinit  --gradinit-alg sgd --gradinit-eta 0.1 --gradinit-gamma 1 --gradinit-normalize-grad --gradinit-lr 1e-2  --gradinit-min-scale 0.01 --gradinit-iters 180 --gradinit-grad-clip 1  
```
- `--gradinit` : Whether to use GradInit. 
- `--gradinit-alg`	 : The target optimization algorithm, deciding the direction of the first gradient step.
- `--gradinit-eta`	: The eta in GradInit.
- `--gradinit-gamma`	: The gradient norm constraint.
- `--gradinit-normalize-grad`	: Number of cycle of pruning that should be done. 
- `--gradinit-lr`	: The learning rate of GradInit.
- `--gradinit-min-scale`	: The lower bound of the scaling factors
- `--gradinit-iters`	: Total number of iterations for GradInit.
- `--gradinit-grad-clip`	: Gradient clipping (per dimension) for GradInit

**The code to support any architecture with only nn.Conv2d, nn.Linear and nn.BatchNorm2d as the parameterized layers. Simply call gradinit_utils.gradinit before your training loop.**


## Acknowledgement
Thanks to Chen Zhu, Renkun Ni, Zheng Xu for opening source of their excellent implementation of GradInit works [GradInit: Learning to Initialize Neural Networks for Stable and Efficient Training](https://github.com/zhuchen03/gradinit?utm_source=catalyzex.com).

## Citation

If you find our code implementation helpful for your own resarch or work, please cite our paper.
```
@inproceedings{jaiswal2022ToST,
  title={Training Your Sparse Neural Network Better with Any Mask},
  author={Jaiswal, Ajay and Ma, Haoyu and Chen, Tianlong and Ding, Ying and Wang, Zhangyang},
  booktitle={International Conference in Machine Learning},
  year={2022}
}
```