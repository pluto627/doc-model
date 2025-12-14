# 🚀 上传模型到 Hugging Face 指南

## 📌 概述

本指南将帮助你把训练好的 Qwen3-VL-30B 医疗模型（V3精准版）上传到 Hugging Face Hub，让其他人可以使用你的模型。

---

## 🔑 第一步：准备 Hugging Face 账号

### 1. 注册账号
- 访问：https://huggingface.co/join
- 注册一个免费账号

### 2. 获取 Access Token
1. 登录后访问：https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 选择 "Write" 权限（允许上传模型）
4. 复制生成的 token（格式类似：`hf_xxxxxxxxxxxxxxxxxxxxx`）

⚠️ **重要**：保管好你的 token，不要分享给别人！

---

## 💻 第二步：安装依赖

```bash
cd /Users/plutoguo/Desktop/training
source venv/bin/activate

# 安装 huggingface_hub
pip install huggingface_hub
```

---

## 🎯 第三步：上传模型

### 方法1：使用自动化脚本（推荐）⭐

```bash
python upload_to_huggingface.py
```

脚本会：
1. ✅ 引导你登录 Hugging Face
2. ✅ 自动创建仓库
3. ✅ 整理所有必要文件
4. ✅ 上传模型和配置
5. ✅ 生成完整的模型卡片

### 方法2：手动上传

#### 2.1 登录 Hugging Face

```bash
huggingface-cli login
```

粘贴你的 access token 并按回车。

#### 2.2 创建仓库

```bash
# 创建一个新的模型仓库
huggingface-cli repo create Qwen3-VL-30B-Medical-V3-Precision --type model
```

仓库名称建议：
- `Qwen3-VL-30B-Medical-V3-Precision` （推荐）
- `Qwen3-Medical-Chinese-V3`
- `医疗咨询模型-V3-精准版`

#### 2.3 上传文件

```bash
# 上传 LoRA 适配器
huggingface-cli upload 你的用户名/Qwen3-VL-30B-Medical-V3-Precision \
  adapters_v3_precision/adapters.safetensors \
  --repo-type model

# 上传配置文件
huggingface-cli upload 你的用户名/Qwen3-VL-30B-Medical-V3-Precision \
  adapters_v3_precision/adapter_config.json \
  --repo-type model

# 上传训练配置
huggingface-cli upload 你的用户名/Qwen3-VL-30B-Medical-V3-Precision \
  finetuned_model_v3_precision/training_config.json \
  --repo-type model

# 上传 README
huggingface-cli upload 你的用户名/Qwen3-VL-30B-Medical-V3-Precision \
  finetuned_model_v3_precision/README.md \
  --repo-type model
```

---

## 📝 第四步：完善模型信息

### 1. 编辑 Model Card（模型卡片）

在 Hugging Face 网页上编辑 `README.md`，添加以下信息：

```markdown
---
language:
- zh
license: apache-2.0
library_name: transformers
tags:
- medical
- chinese
- qwen
- lora
- healthcare
base_model: Qwen/Qwen2-VL-30B
datasets:
- medical-conversations
pipeline_tag: text-generation
---

# Qwen3-VL-30B 医疗咨询模型 V3 - 精准版

## 模型描述

这是一个专门针对中文医疗咨询场景微调的大语言模型，基于 Qwen3-VL-30B。

### 核心特点
- ✅ **极高精准度**：准确的医学术语和数值
- ✅ **温暖人情味**：保持同理心和情绪支持
- ✅ **5200步训练**：充分优化的模型
- ✅ **LoRA 适配器**：轻量级，易于部署

## 使用方法

### 安装依赖

\```bash
pip install mlx mlx-lm
\```

### 加载模型

\```python
from mlx_lm import load, generate

# 加载基础模型和适配器
model, tokenizer = load(
    "Qwen/Qwen2-VL-30B",
    adapter_path="你的用户名/Qwen3-VL-30B-Medical-V3-Precision"
)

# 生成回复
prompt = "我的血压140/90需要担心吗？"
response = generate(model, tokenizer, prompt=prompt, max_tokens=300)
print(response)
\```

### 命令行使用

\```bash
mlx_lm.generate \
  --model Qwen/Qwen2-VL-30B \
  --adapter-path 你的用户名/Qwen3-VL-30B-Medical-V3-Precision \
  --prompt "我的空腹血糖7.2 mmol/L正常吗？" \
  --max-tokens 300
\```

## 训练详情

- **基础模型**: Qwen3-VL-30B Medical V2 Fused
- **训练步数**: 5200
- **LoRA Rank**: 128
- **学习率**: 3e-6
- **批次大小**: 2
- **训练数据**: 4000条医疗对话 + 药物知识增强

### 训练阶段

1. **Phase 1 (0-2000步)**: 精准度核心强化
2. **Phase 2 (2000-4000步)**: 医学知识深化
3. **Phase 3 (4000-5200步)**: 精度+人情味平衡

## 性能指标

| 指标 | 评分 |
|------|------|
| 医学术语准确性 | ★★★★★ |
| 数值精度 | ★★★★★ |
| 诊断置信度 | ★★★★★ |
| 治疗具体性 | ★★★★★ |
| 人情味表达 | ★★★★☆ |

## 许可证

Apache 2.0

## 免责声明

⚠️ 本模型仅供参考，不能替代专业医疗建议。遇到健康问题请咨询专业医生。
```

### 2. 添加标签和元数据

在模型页面右侧添加：
- **Language**: Chinese (zh)
- **License**: Apache 2.0
- **Tags**: medical, chinese, healthcare, qwen, lora
- **Base Model**: Qwen/Qwen2-VL-30B

---

## 🎨 第五步：选择上传什么

### 必须上传的文件 ✅

```
你的模型仓库/
├── adapters.safetensors          # LoRA 权重（最重要！）
├── adapter_config.json            # 适配器配置
├── training_config.json           # 训练配置
└── README.md                      # 模型说明
```

### 可选上传的文件

```
├── metrics_history.json           # 训练曲线数据
├── V3训练使用指南.md              # 使用指南
└── checkpoints/                   # 中间检查点（如果想分享）
    ├── step_2000/
    └── step_4000/
```

### 不建议上传 ❌

```
❌ 基础模型权重（太大，用户应该自己下载 Qwen3-VL-30B）
❌ 日志文件
❌ 虚拟环境
❌ 数据集原始文件
```

---

## 🔍 第六步：验证上传

### 1. 检查模型页面

访问：`https://huggingface.co/你的用户名/Qwen3-VL-30B-Medical-V3-Precision`

确认：
- ✅ 所有文件都已上传
- ✅ README 显示正常
- ✅ 标签和元数据正确

### 2. 测试下载

```bash
# 测试别人能否下载你的模型
huggingface-cli download 你的用户名/Qwen3-VL-30B-Medical-V3-Precision
```

### 3. 测试加载

```python
from mlx_lm import load

model, tokenizer = load(
    "Qwen/Qwen2-VL-30B",
    adapter_path="你的用户名/Qwen3-VL-30B-Medical-V3-Precision"
)
print("✅ 模型加载成功！")
```

---

## 🌟 第七步：分享你的模型

### 1. 设置模型可见性

- **Public**（公开）：所有人都可以看到和使用（推荐）
- **Private**（私有）：只有你可以访问

### 2. 添加示例代码

在 Hugging Face 页面的 "Use this model" 部分添加使用示例。

### 3. 分享链接

你的模型链接：
```
https://huggingface.co/你的用户名/Qwen3-VL-30B-Medical-V3-Precision
```

可以分享到：
- GitHub README
- 社交媒体
- 技术博客
- 论文中

---

## ⚡ 快速上传脚本

我已经为你准备了一个自动化脚本 `upload_to_huggingface.py`，运行它即可：

```bash
python upload_to_huggingface.py
```

脚本会自动：
1. 检查所有必要文件
2. 创建优化的 README
3. 上传所有文件
4. 设置正确的元数据

---

## 🛠️ 故障排查

### 问题1：认证失败

```
Error: Invalid token
```

**解决**：
- 确认 token 有 "Write" 权限
- 重新登录：`huggingface-cli login`

### 问题2：仓库已存在

```
Error: Repository already exists
```

**解决**：
- 使用不同的仓库名
- 或删除旧仓库后重新创建

### 问题3：文件太大

```
Error: File size exceeds limit
```

**解决**：
- 使用 Git LFS：`git lfs install`
- 或分批上传文件

### 问题4：上传速度慢

**解决**：
- 使用国内镜像（如果有）
- 或在网络较好时上传
- 使用 `--resume` 参数断点续传

---

## 📚 更多资源

- [Hugging Face 官方文档](https://huggingface.co/docs/hub/models-uploading)
- [LoRA 适配器说明](https://huggingface.co/docs/peft/conceptual_guides/adapter)
- [模型卡片最佳实践](https://huggingface.co/docs/hub/model-cards)

---

## ✨ 提示

1. **模型命名**：使用清晰的名称，包含模型大小和用途
2. **README 详细**：越详细越好，帮助用户快速上手
3. **添加示例**：提供实际的使用代码
4. **标注限制**：说明模型的适用场景和限制
5. **许可证明确**：选择合适的开源许可证

---

**祝你上传成功！** 🎉

有问题随时查看这份指南或访问 Hugging Face 文档。

