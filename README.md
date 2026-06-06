# BCode

#### 介绍
Research on Multi modal Perception Enhanced Reasoning Detection Framework

#### 补充说明
dataset目录下refcoco，refcoco+，refcocog文件夹，用于对模型进行评估，需要补充下载train2024用来添加对应图片

在model文件夹下创建Depth，facebook，vision_model几个文件夹，分别对应depth anything 3，sam3，clip的模型文件

实际模型训练需要创建data文件夹在根目录下用于存放预训练的数据集，同时在根目录下创建out文件夹用于存放训练权重；

预训练数据集可以直接使用常见中英文问答数据集，参考如<https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V>

#### 代码说明
refcoco_dataset 用于处理数据集中缺失的特殊标记；

model目录下为实际搭建的model文件包括多模态推理模型model_vlm，最终的模型model_vlm_RCoder，对应的分词器等，以及对应的对比模型代码示例model_qwen3_world_rcoder；

trainer为训练代码合集，包括对应的微调代码，训练代码，对应的常见的训练工具；

根目录下为具体的模型测试代码，可以直接替换对应的模型
