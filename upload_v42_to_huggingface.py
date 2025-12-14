#!/usr/bin/env python3
"""
上传 V4.2 完整模型到 HuggingFace
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# 配置
MODEL_PATH = "/Volumes/Pluto/Code/Model/lmstudio-models/local/XunDoc-30B-V42-Final"
REPO_ID = "pluto6272/Qwen3-VL-30B-Medical-V3-Precision"  # 更新现有仓库

# README 内容
README_CONTENT = """---
license: apache-2.0
language:
- zh
- en
tags:
- medical
- healthcare
- chinese
- qwen
- vision
- mlx
pipeline_tag: text-generation
---

# 🏥 XunDoc-30B 医疗咨询模型 V4.2

## 📌 模型描述

这是一个专门针对**中文医疗咨询场景**微调的大语言模型，基于 **Qwen2-VL-30B**。

V4.2 版本在 V4.1 基础上进行了重大改进，重点提升了：
- 🎯 **智能判断**：根据信息充足程度决定是否追问
- 🔄 **回复多样性**：更自然的对话体验
- 📊 **医学准确性**：精准的医学术语和数值
- 💝 **自然人情味**：不机械，更贴心

## 🌟 V4.2 核心改进

| 改进项 | 说明 |
|-------|------|
| ✅ 智能判断 | 信息充足时直接回答，不足时才追问 |
| ✅ 回复多样性 | temperature=0.7，回复更自然 |
| ✅ 准确性提升 | 增强了精准医学知识库 |
| ✅ 自然结尾 | 不再机械地每次都问问题 |
| ✅ 损失下降89% | 从2.331降到0.263 |

## 🚀 快速开始

### 安装依赖

```bash
pip install mlx mlx-lm transformers
```

### 使用模型

#### 方法1: MLX 命令行聊天

```bash
mlx_lm.chat --model pluto6272/XunDoc-30B-Medical-V42
```

#### 方法2: Python API

```python
from mlx_lm import load, generate

# 加载模型
model, tokenizer = load("pluto6272/XunDoc-30B-Medical-V42")

# 生成回复
prompt = "我的血压145/92，早上空腹测的，需要担心吗？"
response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=500,
    temp=0.7  # 增加多样性
)
print(response)
```

#### 方法3: LM Studio

1. 下载模型到本地
2. 在 LM Studio 中添加本地模型
3. 加载并开始使用

## 💬 使用示例

### 示例1: 信息充足 → 直接回答

**输入**：
> 我的血压是145/92，早上空腹测的，之前休息了5分钟，没吃降压药

**输出**：
> 好的，我来帮您分析这个血压数值。
> 
> 📊 **数据解读**：
> - 收缩压 145 mmHg（正常<140）
> - 舒张压 92 mmHg（正常<90）
> - 属于 **1级高血压**
> 
> 您的测量条件很规范（空腹、休息后），数据比较可靠。
> 
> 💡 **建议**：
> 1. 连续监测一周，每天同一时间测量
> 2. 记录数值，观察变化趋势
> 3. 减少盐分摄入（每日<6g）
> 4. 如持续偏高，建议心内科就诊
> 
> 单次测量偏高不必太紧张，持续监测更重要~

### 示例2: 信息不足 → 智能追问

**输入**：
> 我血压有点高

**输出**：
> 血压偏高的情况因人而异，需要了解更多才能给您准确建议。
> 
> 方便告诉我：
> 1. 具体数值是多少？（比如145/90）
> 2. 什么时候测的？（早上/晚上，饭前/饭后）
> 
> 这样我能帮您判断是否需要重视~

## 📊 训练指标

| 指标 | 数值 |
|------|------|
| 初始验证损失 | 2.331 |
| 最终验证损失 | 0.263 |
| 测试损失 | 0.265 |
| 测试困惑度 | 1.303 |
| 损失下降 | **89%** |
| 训练步数 | 7000+ (2000 + 5000) |

## 🎯 训练详情

### 训练数据 (V4.2)

| 数据类型 | 说明 |
|---------|------|
| 现有中文医学问答 | 保持专业性 |
| 图像分析数据 | 提升图像理解 |
| 情境化回复 | 智能判断何时追问 |
| 多轮对话 | 增强上下文理解 |
| 精准医学知识 | 准确的数值和术语 |
| 身份认知 | 保持一致身份 |

### 技术细节

- **基础模型**: Qwen2-VL-30B
- **微调方法**: LoRA
- **训练框架**: MLX
- **学习率**: 2e-6
- **批次大小**: 1-2
- **LoRA层数**: 16

## 🔧 推荐生成参数

```python
generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512,
    temp=0.7,                    # 增加多样性
    top_p=0.9,                   # 核采样
    repetition_penalty=1.1,      # 减少重复
)
```

## ⚠️ 使用限制与免责声明

### ✅ 适用场景
- 一般健康咨询与科普
- 检查结果的初步解读
- 用药常识与注意事项
- 健康生活方式建议

### ❌ 不适用场景
- 紧急医疗情况（请立即就医或拨打120）
- 最终诊断判断（需要专业医生面诊）
- 处方药物开具（需要医生处方）

### ⚠️ 重要免责声明

**本模型仅供参考学习，不能替代专业医疗建议。**

- AI模型可能产生不准确或错误的信息
- 不应作为医疗决策的唯一依据
- 任何健康问题都应咨询专业医生
- 紧急情况请立即就医或拨打急救电话

## 📄 许可证

本模型基于 **Apache 2.0** 许可证开源。

## 🙏 致谢

- **Qwen 团队**: 提供优秀的 Qwen2-VL-30B 基础模型
- **MLX 团队**: Apple Silicon 深度学习优化框架

---

**模型版本**: V4.2  
**更新日期**: 2024年12月  
**开发者**: pluto6272
"""

def main():
    print("=" * 60)
    print("🚀 上传 XunDoc-30B-V42 到 HuggingFace")
    print("=" * 60)
    
    # 检查模型路径
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        return
    
    print(f"\n📁 模型路径: {MODEL_PATH}")
    print(f"📦 目标仓库: {REPO_ID}")
    
    # 写入 README
    readme_path = model_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(README_CONTENT)
    print(f"✅ 已更新 README.md")
    
    # 初始化 API
    api = HfApi()
    
    # 创建仓库（如果不存在）
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ 仓库已准备: {REPO_ID}")
    except Exception as e:
        print(f"⚠️ 仓库创建提示: {e}")
    
    # 上传文件夹
    print("\n📤 开始上传（这可能需要较长时间，模型约16GB）...")
    print("   请耐心等待...\n")
    
    try:
        upload_folder(
            folder_path=MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload XunDoc-30B-V42 complete model",
            ignore_patterns=["*.pyc", "__pycache__", ".DS_Store"]
        )
        
        print("\n" + "=" * 60)
        print("✅ 上传完成！")
        print("=" * 60)
        print(f"\n🔗 模型地址: https://huggingface.co/{REPO_ID}")
        print("\n📥 其他人可以这样使用:")
        print(f"   mlx_lm.chat --model {REPO_ID}")
        
    except Exception as e:
        print(f"\n❌ 上传失败: {e}")
        print("\n请确保:")
        print("1. 已登录 HuggingFace: huggingface-cli login")
        print("2. 有足够的网络带宽")
        print("3. HuggingFace token 有写入权限")

if __name__ == "__main__":
    main()

