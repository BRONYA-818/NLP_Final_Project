# NLP Final Project,AIA,HUST,2024
本次作业旨在实践图像描述任务,掌握使用多模态大模型进行 prompt engineering 的技能,同时体验使用 CNN-RNN 等模型架构完成图像到文本的生成任务。通过本次作业,将对多模态学习的基本流程和方法有一个全面的认识。
## 小组成员
队长：梁一凡 本硕博2101班U202115210 40%\
成员：祝家心 本硕博2101班U202115191 20%\
成员：邓　斌 智　实2101班U202115228 30%\
成员：黄骏言 智　实2101班U202115263 10%

## 指导老师
刘禹良、陈伟

## 项目简介
在本研究中，我们使用了阿里云的 Qwen-VL多模态大模型对未标注的图片进行描述生成。主要任务是创造多维度的描述，以增强对视觉内容的理解。我们通过提示工程技术（Prompt Engineering）来优化输入查询（即提示），以提高模型输出的质量和相关性。随后我们使用生成的描述作为标签训练 MiniCPM-V 模型，为了提高训练速度和节省内存，我们以预训练模型为基础分别进行 LoRA 微调和全量微调，采用了 BLEU等五种指标评估生成的图像描述的质量，并对比分析了预训练模型和不同微调技术训练后的模型性能。我们将模型进行了 web 对话部署，并使用自有数据集的图片测试了模型的泛化能力。

## 创建虚拟环境
要创建并激活虚拟环境，请按照以下步骤操作：
### 克隆仓库
首先，克隆本仓库到你的本地机器：
```bash
git clone https://github.com/BRONYA-818/NLP_Final_Project.git
cd your-repo
```
### 使用 `conda`
```bash
conda env create -f environment.yaml
conda activate MiniCPMV
```

## 关于Huggingface下载问题
由于我们的项目可能需要从Huggingface上下载文件，对于连接问题有以下解决方案：

1.科学上网\
2.使用镜像站
* 如果是linux，在终端运行
```bash
export HF_ENDPOINT="https://hf-mirror.com" 
```
* 如果是windows,在终端运行
```cmd
set HF_ENDPOINT="https://hf-mirror.com"
```
* 在.py文件开头增加
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```


## 数据与模型权重准备
请将数据集(包括图片文件夹和json标注文件）放置在：
```
finetune/data
```
最后该文件夹应该有如下文件:
```
finetune/
└── data/
    ├── Train/
    │   ├── 0000.jpg
    │   ├── 0001.jpg
    │   └── ...
    ├── Val/
    │   ├── 2002.jpg
    │   └── ...
    ├── train.json
    └── eval.json
```
模型权重请在[百度网盘链接（提取码：jg59 ）](https://pan.baidu.com/s/1_cCNee6uniTtP3c3ZUcgJA?pwd=jg59 )下载，位于NLP Final Project/MiniCPM-V/LoRA权重/output,请将output文件夹置于finetune目录下。因文件较大，可以只下载
```
checkpoint-5000
adapter_model.safetensors
tokenizer_config
trainer_state.json
training_args.bin
adapter_config.json
special_tokens_map.json
tokenizer.model
```
最后文件组织形式为
```
finetune/ 
└── output/ 
    ├── checkpoint-5000/ 
    ├── adapter_model.safetensors 
    ├── tokenizer_config 
    ├── trainer_state.json 
    ├── training_args.bin 
    ├── adapter_config.json 
    ├── special_tokens_map.json 
    └── tokenizer.model
```
## 复现训练
训练脚本
```
finetune/finetune_ds.sh（用于全量微调）
finetune/finetune_lora.sh（用于LoRA微调）
```
在finetune目录下运行
```bash
bash finetune_ds.sh 
```
或
```bash
bash finetune_lora.sh
```
 即可复现训练，更多与训练有关的细节请查看```finetune/readme.md```，里面介绍了如何准备数据、如何微调数据、运行配置的要求以及其他常见问题。

**关于运行配置要求**

以下表格展示了在使用 NVIDIA A100 (80GiB) GPU 进行微调时，模型在不同数量的 GPU 下的内存使用情况。微调使用了 DeepSpeed Zero-3 优化、梯度检查点技术以及将优化器和参数内存卸载到 CPU，最大长度设置为 2048，批量大小设置为 1。


|微调策略|GPUs: 2 |GPUs: 4|GPUs: 8|
|-------------------------|--------------------------|-------------------------|-------------------------|
|LoRA微调|14.4 GiB |13.6 GiB|13.1 GiB|
|全量微调|16.0 GiB |15.8 GiB|15.63GiB|



## 推理及webUI 对话
推理请运行：
```bash
python inference.py
```
webUI 对话demo请运行
```bash
python web_demo.py
```
指标计算请运行
```bash
python metrics.py
```
运行推理和webUI demo时请注意选择模型类型和模型路径，所有代码运行时也需要注意根据自己的路径进行修改
```python
type="pretrained"  #在这里进行修改，以决定使用的模型类型
assert type in ["pretrained", "ds", "lora"]
model_path = 'openbmb/MiniCPM-V-2'#预训练模型路径
# model_path = 'finetune/output/xxx.pt'#全量微调或LoRA微调模型路径
```
如果受限于硬件条件无法运行webUI demo，可以联系QQ：879059433，我将在本地服务器上运行并开放链接供测试使用。如果有其他任何问题也可以提issue或者直接QQ联系。
以下是webUI demo的演示图片：
<img alt="webUI" src="/data/MiniCPM-V/figure/webUI demo.jpeg" width=400>



## 贡献
如果你想为本项目做出贡献，请遵循以下步骤：

1. Fork 本仓库。
2. 创建一个新的分支：
```bash
git checkout -b feature-branch
```
3. 提交你的更改：
```bash
git commit -am '添加了新功能'
```
4. 推送到分支：
```bash
git push origin feature-branch
```
5. 创建一个新的 Pull Request。

## 许可证
本项目使用的许可证类型。详细信息请参见 [LICENSE](./LICENSE) 文件。

## 致谢
感谢以下开源项目的工作，使本项目得以实现：

- [Qwen-VL](https://github.com/username/repo3)
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) 
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 
 

## 特别感谢
<img alt="洛天依" src="/data/MiniCPM-V/figure/lty.png" width=200>\
感谢阿八的歌声陪伴我每一个星夜\
[洛天依B站官号](https://space.bilibili.com/36081646?spm_id_from=333.337.0.0)\
关注洛天依喵谢谢喵\
点个关注请你吃小笼包\
不关注的人都坏坏喵
