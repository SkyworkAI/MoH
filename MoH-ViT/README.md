<div align=center>
<img src="../figures/fig1.png" width="280px">
</div>

<h2 align="center"> <a href="">MoH: Multi-Head Attention as Mixture-of-Head Attention

</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>


# ImageNet-1K classification with MoH-ViT

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
```bash
pip install -r requirements.txt
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html), and the training and validation data is expected to be in the `train` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

### Evaluation

To evaluate the pre-trained MoH-ViT on ImageNet-1K val with GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=2024 \
--use_env main.py \
--config ./configs/${MODEL_TYPE}.py \
--data-path ${ImageNet-1K_PATH} \
--resume ./checkpoints/${MODEL_TYPE}.pth \
--eval
```

### ImageNet-1K Training

To train MoH-ViT on ImageNet-1K using 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=2024 \
--use_env main.py \
--config ./configs/${MODEL_TYPE}.py \
--data-path ${ImageNet-1K_PATH} \
--batch-size 128 \
--output_dir results/${MODEL_TYPE} \
--num_workers 32
```

or

```bash
bash moh_transnext_base_75.sh ${ImageNet-1K_PATH}
bash moh_transnext_base_50.sh ${ImageNet-1K_PATH}
bash moh_transnext_small_80.sh ${ImageNet-1K_PATH}
bash moh_transnext_small_75.sh ${ImageNet-1K_PATH}
```