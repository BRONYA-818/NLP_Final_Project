#!/usr/bin/env python
# encoding: utf-8
import os
import json
from tqdm import tqdm
from peft import PeftModel
import gradio as gr
from PIL import Image
import traceback
import re
import torch
import argparse
from transformers import AutoModel, AutoTokenizer

# README, How to run demo on different devices
# For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
# python web_demo.py --device cuda --dtype bf16

# For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
# python web_demo.py --device cuda --dtype fp16

# For Mac with MPS (Apple silicon or AMD GPUs).
# PYTORCH_ENABLE_MPS_FALLBACK=1 python web_demo.py --device mps --dtype fp16

# Argparser
ERROR_MSG = "Error, please retry"
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--device', type=str, default='cuda', help='cuda or mps')
parser.add_argument('--dtype', type=str, default='bf16', help='bf16 or fp16')
args = parser.parse_args()
device = args.device
assert device in ['cuda', 'mps']
if args.dtype == 'bf16':
    if device == 'mps':
        print('Warning: MPS does not support bf16, will use fp16 instead')
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
else:
    dtype = torch.float16
    
def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
    default_params = {"num_beams":3, "repetition_penalty": 1.2, "max_new_tokens": 1024}
    if params is None:
        params = default_params
    if img is None:
        return -1, "Error, invalid image, please upload a new image", None, None
    try:
        image = img.convert('RGB')
        answer, context, _ = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            **params
        )
        res = re.sub(r'(<box>.*</box>)', '', answer)
        res = res.replace('<ref>', '')
        res = res.replace('</ref>', '')
        res = res.replace('<box>', '')
        answer = res.replace('</box>', '')
        return 0, answer, None, None
    except Exception as err:
        print(err)
        traceback.print_exc()
        return -1, ERROR_MSG, None, None

# Load model
# 如果要使用没有微调的模型
model_path = 'openbmb/MiniCPM-V-2'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device=device, dtype=dtype)
model.eval()

# 如果要使用LoRA微调的模型
# model_type=  "openbmb/MiniCPM-V-2" 
# path_to_adapter="finetune/output/output_lora/checkpoint-5000"
# model =  AutoModel.from_pretrained(
#         model_type,
#         trust_remote_code=True
#         )
# tokenizer = AutoTokenizer.from_pretrained(path_to_adapter, trust_remote_code=True)
# lora_model = PeftModel.from_pretrained(
#     model,
#     path_to_adapter,
#     device_map="auto",
#     trust_remote_code=True
# ).eval().cuda()

# 遍历目录并进行推理
input_dir = 'finetune/data/Val/'
output_file = 'result/pt/output.json'
results = []

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 使用 tqdm 添加进度条，并对文件名进行排序
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
for idx, filename in enumerate(tqdm(image_files, desc="Processing images")):
    image_path = os.path.join(input_dir, filename)
    image = Image.open(image_path).convert('RGB')
    question = "请告诉我这张照片记录了一个什么场景，尽可能包含图像中出现的所有物体。"
    msgs = [{'role': 'user', 'content': question}]
    _, answer, _, _ = chat(image, msgs, None)
    
    result = {
        "id": filename.split('.')[0],  # 使用文件名作为 id
        "image": image_path,
        "conversations": [
            {
                "role": "user",
                "content": "<image>\n" + question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }
    results.append(result)

# 将结果写入 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file}")