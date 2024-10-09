<div align=center>
<img src="../figures/fig1.png" width="280px">
</div>

<h2 align="center"> <a href="">MoH: Multi-Head Attention as Mixture-of-Head Attention

</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>


# Continue-Tuning LLaMA3-8B

### 💡 Download URL

<div align=center>

|                   Code                    |                                                                                                                         HuggingFace Model                                                                                                                         |  
|:-----------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     **[MoH-ViT](MoH-ViT/Readme.md)**      | 🤗 [MoH-ViT-B-75](https://huggingface.co/Chat-UniVi/MoH-ViT-B-75), [MoH-ViT-B-50](https://huggingface.co/Chat-UniVi/MoH-ViT-B-50), [MoH-ViT-S-80](https://huggingface.co/Chat-UniVi/MoH-ViT-S-80), [MoH-ViT-S-75](https://huggingface.co/Chat-UniVi/MoH-ViT-S-75) |
|     **[MoH-DiT](MoH-DiT/Readme.md)**      |                                                                                                 😊 [MoH-DiT-90](https://huggingface.co/Chat-UniVi/MoH-DiT-XL-90)                                                                                                  | 
| **[MoH-LLaMA3-8B](MoH-LLaMA3/Readme.md)** |                                                                                                                        😊 [MoH-LLaMA3-8B](https://huggingface.co/Chat-UniVi/MoH-LLaMA3-8B)                                                                                                                         | 

</div>


## 🤖 API for Model Inference
If you want to load the model from the model hub on Hugging Face or on local, you can use the following code snippets.

### Base Model Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

question = "Hello!"

model = AutoModelForCausalLM.from_pretrained("Chat-UniVi/MoH-LLaMA3-8B", trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("Chat-UniVi/MoH-LLaMA3-8B", trust_remote_code=True)

inputs = tokenizer(question, return_tensors='pt').to(model.device)
response = model.generate(inputs.input_ids, max_length=128)
print(tokenizer.decode(response.cpu()[0], skip_special_tokens=True))
```

### Chat Model Inference
Coming soon...


## 🗝️ Training & Validating
* The training code is built on [Skywork-MoE](https://github.com/SkyworkAI/Skywork-MoE). Unless Skywork-MoE is open source, we can't open source MoH-LLaMA3 alone. We will release the training code after the approval is completed.
* The evaluation is performed on multiple key benchmarks using the [Eleuther AI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

```python
# For example, test MoH-LLaMA3-8B on winogrande

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--main_process_port 2004 -m lm_eval --model hf \
--model_args pretrained=Chat-UniVi/MoH-LLaMA3-8B \
--tasks winogrande \
--batch_size 1 \
--output_path Results/winogrande
```