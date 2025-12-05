#!/bin/bash
# 最终版训练脚本 - 精度优先

set -e

echo "=================================================="
echo "🎯 医疗VLM最终训练 - 精度优先版"
echo "=================================================="
echo ""
echo "优先级顺序："
echo "  1. 精度 (50%) 🥇"
echo "  2. 人情味 (35%) 🥈"
echo "  3. 图像识别 (15%) 🥉"
echo ""
echo "=================================================="

# 进入项目目录
cd "$(dirname "$0")"

# 激活虚拟环境
if [ -d "venv" ]; then
    echo "📦 激活虚拟环境..."
    source venv/bin/activate
fi

# 检查数据
if [ ! -f "data/processed/train.jsonl" ]; then
    echo "❌ 训练数据不存在！"
    echo "请先运行: python preprocess_data.py"
    exit 1
fi

# 基础模型路径
BASE_MODEL="/Users/plutoguo/.lmstudio/models/lmstudio-community/Qwen3-VL-30B-Medical-Finetuned"

if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ 基础模型不存在: $BASE_MODEL"
    exit 1
fi

echo ""
echo "🚀 开始训练..."
echo ""

# 运行训练
python train_final.py \
  --base-model "$BASE_MODEL" \
  --steps 2000 \
  --batch-size 2 \
  --lr 5e-6 \
  --lora-rank 128 \
  --accuracy-reward 2.0 \
  --empathy-reward 1.0 \
  --vision-reward 0.6 \
  --accuracy-penalty 3.0 \
  --coldness-penalty 1.5

echo ""
echo "=================================================="
echo "🎉 训练完成！"
echo "=================================================="
echo "模型保存于: finetuned_model_final/"
echo "检查点保存于: checkpoints_final/"
echo "日志保存于: logs/"
echo ""

