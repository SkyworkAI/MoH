<div align=center>
<img src="../figures/fig1.png" width="280px">
</div>

<h2 align="center"> <a href="">MoH: Multi-Head Attention as Mixture-of-Head Attention

</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align=center>

<!-- [![Demo](https://img.shields.io/badge/⚡-Hugging%20Face%20Demo-yellow.svg)](https://huggingface.co/spaces/Chat-UniVi/Chat-UniVi) -->
[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/Chat-UniVi)
[![arXiv](https://img.shields.io/badge/Arxiv-2311.08046-b31b1b.svg?logo=arXiv)]()
[![License](https://img.shields.io/badge/Code%20License-Apache2.0-yellow)](https://github.com/SkyworkAI/MoH/blob/main/LICENSE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FSkyworkAI%2FMoH&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/SkyworkAI/MoH?color=critical&label=Issues)](https://github.com/SkyworkAI/MoH/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/SkyworkAI/MoH?color=success&label=Issues)](https://github.com/SkyworkAI/MoH/issues?q=is%3Aissue+is%3Aclosed)
</h5>

# Class-Conditional Image Generation with MoH-DiT

### 💡 Download URL

<div align=center>

|                   Code                    |                                                                                                                         HuggingFace Model                                                                                                                         |  
|:-----------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     **[MoH-ViT](https://github.com/SkyworkAI/MoH/tree/main/MoH-ViT)**      | 🤗 [MoH-ViT-B-75](https://huggingface.co/Chat-UniVi/MoH-ViT-B-75), [MoH-ViT-B-50](https://huggingface.co/Chat-UniVi/MoH-ViT-B-50), [MoH-ViT-S-80](https://huggingface.co/Chat-UniVi/MoH-ViT-S-80), [MoH-ViT-S-75](https://huggingface.co/Chat-UniVi/MoH-ViT-S-75) |
|     **[MoH-DiT](https://github.com/SkyworkAI/MoH/tree/main/MoH-DiT)**      |                                                                                                 😊 [MoH-DiT-90](https://huggingface.co/Chat-UniVi/MoH-DiT-XL-90)                                                                                                  | 
| **[MoH-LLaMA3-8B](https://github.com/SkyworkAI/MoH/tree/main/MoH-LLaMA3)** |                                                                                                                        😊 [MoH-LLaMA3-8B](https://huggingface.co/Chat-UniVi/MoH-LLaMA3-8B)                                                                                                                         | 

</div>

## 🛠️ Requirements and Installation
### Requirements

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate DiT
```


### Sampling 

If you've trained a new MoH-DiT model with [`train.py`](train.py) (see [below](#training-dit)), you can add the `--ckpt`
argument to use your own checkpoint instead. For example, to sample from the EMA weights of a custom 
256x256 MoH-DiT-XL/2-90 model, run:

```bash
python sample.py --model MoH-DiT-XL/2-90 --image-size 256 --ckpt /path/to/model.pt
```


### Training MoH-DiT

We provide a training script for MoH-DiT in [`train.py`](train.py). This script can be used to train class-conditional 
MoH-DiT models, but it can be easily modified to support other types of conditioning. To launch MoH-DiT-XL/2-90 (256x256) training with 8 GPUs on 
one node:

```bash
torchrun --nnodes=1 \
--nproc_per_node=8 train.py \
--model MoH-DiT-XL/2-90 \
--data-path /path/to/imagenet/train \
--results-dir results/MoH-DiT-XL-2-90
```

## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a MoH-DiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained MoH-DiT-XL/2-90 model over 8 GPUs, run:

```bash
torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py --model MoH-DiT-XL/2-90 --num-fid-samples 50000
```

There are several additional options; see [`sample_ddp.py`](sample_ddp.py) for details.
