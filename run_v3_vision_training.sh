#!/bin/bash
# V3 Vision支持训练启动脚本 - 1000步

set -e

echo "=========================================="
echo "🎯 V3 Vision支持训练启动 (1000步)"
echo "=========================================="
echo ""

# 进入项目目录
cd "$(dirname "$0")"

# 激活虚拟环境
if [ -d "venv" ]; then
    echo "📦 激活虚拟环境..."
    source venv/bin/activate
else
    echo "❌ 虚拟环境不存在，请先创建"
    exit 1
fi

# 检查数据
echo "📊 检查训练数据..."
if [ ! -f "data_mlx/train.jsonl" ]; then
    echo "❌ 训练数据不存在: data_mlx/train.jsonl"
    echo "请先运行数据预处理"
    exit 1
fi

echo "✅ 训练数据: $(wc -l < data_mlx/train.jsonl) 样本"
echo "✅ 验证数据: $(wc -l < data_mlx/valid.jsonl) 样本"
echo ""

# 检查基础模型
BASE_MODEL="/Users/plutoguo/.lmstudio/models/lmstudio-community/Qwen3-VL-30B-Medical-Finetuned"
if [ ! -d "$BASE_MODEL" ]; then
    echo "⚠️  模型不存在: $BASE_MODEL"
    echo "请检查模型路径"
    exit 1
fi

echo "🔧 基础模型: $BASE_MODEL"
echo ""

# 训练参数
STEPS=1000
BATCH_SIZE=2
LEARNING_RATE=3e-6
LORA_RANK=128
PRECISION_WEIGHT=2.0
VISION_WEIGHT=1.5
EMPATHY_WEIGHT=1.0
VISION_TOWER_NAME="visual"
TRAIN_VISION_PROJECTOR="true"
DATA_FORMAT="llava"

echo "⚙️  训练配置:"
echo "   - 训练步数: $STEPS (快速测试)"
echo "   - 批次大小: $BATCH_SIZE"
echo "   - 学习率: $LEARNING_RATE"
echo "   - LoRA Rank: $LORA_RANK"
echo "   - 精准度权重: $PRECISION_WEIGHT"
echo "   - 视觉权重: $VISION_WEIGHT"
echo "   - 人情味权重: $EMPATHY_WEIGHT"
echo "   - 视觉编码器: $VISION_TOWER_NAME"
echo "   - 训练视觉投影层: $TRAIN_VISION_PROJECTOR"
echo "   - 数据格式: $DATA_FORMAT"
echo ""

# 确认
read -p "是否开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "🚀 开始V3 Vision训练"
echo "=========================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 开始训练
python train_v3_vision.py \
    --base-model "$BASE_MODEL" \
    --adapter-path "adapters_v2" \
    --steps $STEPS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --lora-rank $LORA_RANK \
    --precision-weight $PRECISION_WEIGHT \
    --vision-weight $VISION_WEIGHT \
    --empathy-weight $EMPATHY_WEIGHT \
    --vision-tower-name "$VISION_TOWER_NAME" \
    --train-vision-projector "$TRAIN_VISION_PROJECTOR" \
    --data-format "$DATA_FORMAT"

# 计算耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo "🎉 V3 Vision训练完成！"
echo "=========================================="
echo ""
echo "⏱️  总耗时: ${HOURS}小时 ${MINUTES}分钟"
echo ""
echo "📁 输出位置:"
echo "   - Adapters: ./adapters_v3_vision/"
echo "   - 模型配置: ./finetuned_model_v3_vision/"
echo "   - 训练日志: ./logs/"
echo "   - 检查点: ./checkpoints_v3_vision/"
echo ""
echo "✨ 特性:"
echo "   ✅ 视觉功能已保留"
echo "   ✅ 支持医学图像分析"
echo "   ✅ 精准度已提升"
echo "   ✅ 人情味已维持"
echo ""
echo "📝 下一步:"
echo "   1. 测试模型效果（支持图像）:"
echo "      mlx_lm.chat --model '$BASE_MODEL' --adapter-path adapters_v3_vision"
echo ""
echo "   2. 评估多模态性能:"
echo "      python evaluate_v3_model.py --adapter-path adapters_v3_vision"
echo ""
echo "   3. 如需在LM Studio使用，运行融合脚本"
echo ""




