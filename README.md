# 🏥 医疗视觉语言模型微调训练项目

## 项目概述

本项目用于微调 Qwen3-VL-30B 视觉语言模型，使其在医疗领域回答更精准，同时具备人情味的交流方式。

## 数据集来源

- **AIREADI Clinical Lab Tests** - 临床实验室检测数据
- **Medical Vision LLM Dataset** - 医疗视觉问答数据
- **Medical Vision Dataset** - 医疗影像数据
- **MedTrinity-25M** - 大规模医疗三元组数据

## 训练特点

1. **监督式学习** - 使用高质量医疗问答对进行训练
2. **惩罚机制** - 对错误回答和不当表达进行惩罚
3. **人情味增强** - 训练模型表达同理心和关怀
4. **10000+ 训练步骤** - 确保充分学习

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
python download_datasets.py
```

### 3. 预处理数据

```bash
python preprocess_data.py
```

### 4. 开始训练

```bash
python train.py
```

### 5. 测试模型

```bash
python evaluate.py
```

## 项目结构

```
training/
├── requirements.txt          # 依赖配置
├── download_datasets.py      # 数据下载脚本
├── preprocess_data.py        # 数据预处理脚本
├── train.py                  # 训练主脚本
├── evaluate.py               # 评估测试脚本
├── config.py                 # 配置文件
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据
│   └── processed/            # 处理后数据
├── checkpoints/              # 模型检查点
└── logs/                     # 训练日志
```

## 硬件要求

- Apple Silicon Mac (M1/M2/M3)
- 建议 32GB+ 统一内存
- 100GB+ 可用存储空间

## 训练配置

- **LoRA Rank**: 64
- **Learning Rate**: 1e-5
- **Batch Size**: 4
- **训练步数**: 10000+
- **惩罚系数**: 0.1

## 联系方式

如有问题，请联系开发者。

