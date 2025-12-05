#!/bin/bash
# 测试训练好的医疗模型

cd "$(dirname "$0")"
source venv/bin/activate

MODEL_PATH="/Users/plutoguo/.lmstudio/models/lmstudio-community/Qwen3-VL-30B-Medical-Finetuned"

echo "🏥 医疗VLM模型测试"
echo "================================"
echo ""

# 读取用户输入
echo "请输入您的医疗问题（或按Ctrl+C退出）："
read -r QUESTION

echo ""
echo "🤖 模型回复："
echo "================================"

mlx_lm.generate \
    --model "$MODEL_PATH" \
    --prompt "$QUESTION" \
    --max-tokens 400 \
    --temp 0.7

echo ""
echo "================================"

