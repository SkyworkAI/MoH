<div align=center>
<img src="../figures/fig1.png" width="280px">
</div>

<h2 align="center"> <a href="">MoH: Multi-Head Attention as Mixture-of-Head Attention

</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>


# Class-Conditional Image Generation with MoH-DiT

### 💡 Download URL

<div align=center>

|                   Code                    |                                                                                                                         HuggingFace Model                                                                                                                         |  
|:-----------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     **[MoH-ViT](MoH-ViT/Readme.md)**      | 🤗 [MoH-ViT-B-75](https://huggingface.co/Chat-UniVi/MoH-ViT-B-75), [MoH-ViT-B-50](https://huggingface.co/Chat-UniVi/MoH-ViT-B-50), [MoH-ViT-S-80](https://huggingface.co/Chat-UniVi/MoH-ViT-S-80), [MoH-ViT-S-75](https://huggingface.co/Chat-UniVi/MoH-ViT-S-75) |
|     **[MoH-DiT](MoH-DiT/Readme.md)**      |                                                                                                 😊 [MoH-DiT-90](https://huggingface.co/Chat-UniVi/MoH-DiT-XL-90)                                                                                                  | 
| **[MoH-LLaMA3-8B](MoH-LLaMA3/Readme.md)** |                                                                                                                        😊 [MoH-LLaMA3-8B](https://huggingface.co/Chat-UniVi/MoH-LLaMA3-8B)                                                                                                                         | 

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