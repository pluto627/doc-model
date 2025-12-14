# Qwen3-VL-30B 医疗模型 - V3精准度强化版

## 🎯 模型信息

- **版本**: V3 Precision Enhanced
- **基础模型**: Qwen3-VL-30B-Medical-V2-Fused
- **训练类型**: 精准度强化训练
- **训练步数**: 5200
- **训练日期**: 2025-12-06

## ✨ 核心优化

### 主要目标
1. **精准度大幅提升** (权重: 2.0)
   - 医学术语准确性 ↑
   - 数值精度 ↑
   - 诊断置信度 ↑
   - 治疗方案具体性 ↑

2. **人情味保持不变** (权重: 1.0)
   - 同理心表达 ✓
   - 温暖语气 ✓
   - 支持性回复 ✓

### 训练阶段
- **Phase 1 (0-2000步)**: 精准度核心强化
  - 重点: 医学术语、数值精度
  - 精准度权重 × 1.5
  
- **Phase 2 (2000-4000步)**: 医学知识深化
  - 重点: 诊断置信度、治疗方案
  - 精准度权重 × 1.3
  
- **Phase 3 (4000-5200步)**: 精度+人情味平衡
  - 重点: 综合平衡调优
  - 精准度与人情味并重

## 📊 技术配置

- **LoRA配置**:
  - Rank: 128
  - Alpha: 256
  - Dropout: 0.05
  - Layers: 32

- **训练参数**:
  - Batch Size: 2
  - Learning Rate: 3e-06
  - Warmup Steps: 200
  - Max Seq Length: 2048

## 🚀 使用方法

### 方式1: 命令行（最快）
```bash
cd /Users/plutoguo/Desktop/training
source venv/bin/activate
mlx_lm.chat --model /Users/plutoguo/.lmstudio/models/local/Qwen3-VL-30B-Medical-V2-Fused --adapter-path ./adapters_v3_precision
```

### 方式2: 融合后在LM Studio使用
需要先运行融合脚本:
```bash
python fuse_v3_model.py
```

然后在LM Studio中加载融合后的模型。

## 📈 相比V2的改进

| 维度 | V2 | V3 (目标) |
|------|-----|-----------|
| 医学术语准确性 | ★★★★☆ | ★★★★★ |
| 数值精度 | ★★★☆☆ | ★★★★★ |
| 诊断置信度 | ★★★★☆ | ★★★★★ |
| 治疗具体性 | ★★★☆☆ | ★★★★★ |
| 人情味 | ★★★★☆ | ★★★★☆ |

## 📝 训练数据

- 训练样本: 4000条
- 验证样本: 500条
- 数据类型: 医疗多模态（文本+图像）

## 💡 使用建议

V3模型特别适合:
- 需要高精度医疗咨询的场景
- 需要明确数值和范围的诊断
- 需要具体治疗方案的情况
- 医学影像分析和OCR识别

同时保持:
- 温暖的交流语气
- 基本的同理心表达
- 对患者的支持和安慰

## 📁 文件结构

```
finetuned_model_v3_precision/
├── training_config.json    # 训练配置
├── metrics_history.json    # 训练指标历史
└── README.md              # 本文件

./adapters_v3_precision/
├── adapters.safetensors   # LoRA权重
└── adapter_config.json    # Adapter配置
```

## 🔗 相关文件

- 训练日志: logs/training_v3_precision_1765003294.log
- 检查点: ./checkpoints_v3_precision/
- 基础模型: /Users/plutoguo/.lmstudio/models/local/Qwen3-VL-30B-Medical-V2-Fused

---

**训练完成时间**: 2025-12-06 14:42:38
