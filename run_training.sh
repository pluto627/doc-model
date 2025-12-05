#!/bin/bash
# 医疗视觉语言模型训练完整流程脚本

set -e

echo "=================================================="
echo "🏥 医疗视觉语言模型训练流程"
echo "=================================================="

# 进入项目目录
cd "$(dirname "$0")"

# 1. 安装依赖
echo ""
echo "📦 步骤 1: 安装依赖..."
pip install -r requirements.txt --quiet

# 2. 创建目录结构
echo ""
echo "📁 步骤 2: 创建目录结构..."
python -c "from config import create_directories; create_directories()"

# 3. 下载数据集
echo ""
echo "📥 步骤 3: 下载医疗数据集..."
python download_datasets.py

# 4. 预处理数据
echo ""
echo "🔧 步骤 4: 预处理数据..."
python preprocess_data.py

# 5. 开始训练
echo ""
echo "🚀 步骤 5: 开始训练 (10000+ 步)..."
python train.py --steps 10000 --batch-size 4 --lr 1e-5 --lora-rank 64 --penalty 0.1

# 6. 评估模型
echo ""
echo "📊 步骤 6: 评估模型..."
python evaluate.py --report evaluation_report.md

echo ""
echo "=================================================="
echo "🎉 训练完成!"
echo "=================================================="
echo "模型保存于: finetuned_model/"
echo "评估报告: evaluation_report.md"
echo ""
echo "运行交互式测试: python evaluate.py --interactive"

