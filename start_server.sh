#!/bin/bash
# 启动医疗图像分析Web服务器

cd "$(dirname "$0")"

echo "=================================================="
echo "🏥 医疗图像分析Web应用启动脚本"
echo "=================================================="

# 检查虚拟环境
if [ -d "venv" ]; then
    echo "✅ 激活虚拟环境..."
    source venv/bin/activate
else
    echo "⚠️  未找到虚拟环境，使用系统Python"
fi

# 安装Flask（如果需要）
echo "📦 检查依赖..."
pip install flask pillow --quiet 2>/dev/null || true

# 启动服务器
echo ""
echo "🚀 启动Web服务器..."
echo ""

python3 app.py



