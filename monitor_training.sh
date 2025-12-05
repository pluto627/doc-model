#!/bin/bash
# 实时监控训练进度

cd "$(dirname "$0")"

LOG_FILE=$(ls -t training_real_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ 未找到训练日志文件"
    exit 1
fi

echo "=================================================="
echo "🎯 医疗VLM训练监控"
echo "=================================================="
echo "日志文件: $LOG_FILE"
echo ""

# 检查进程
PID=$(ps aux | grep mlx_lm.lora | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ 训练进程未运行"
    echo ""
    echo "最后50行日志："
    tail -50 "$LOG_FILE"
    exit 1
fi

echo "✅ 训练进程正在运行 (PID: $PID)"
echo ""

# 显示最新进度
echo "📊 最新训练进度："
echo "=================================================="
tail -50 "$LOG_FILE" | grep -E "Iter [0-9]+:" | tail -10
echo "=================================================="
echo ""

# 内存使用
MEM=$(ps aux | grep mlx_lm.lora | grep -v grep | awk '{print $4}')
echo "💾 内存使用: ${MEM}%"
echo ""

# 统计信息
TOTAL_ITERS=$(grep "iters:" "$LOG_FILE" | head -1 | awk -F'iters: ' '{print $2}' | awk '{print $1}')
CURRENT_ITER=$(tail -100 "$LOG_FILE" | grep -E "Iter [0-9]+:" | tail -1 | awk '{print $2}' | tr -d ':')

if [ ! -z "$TOTAL_ITERS" ] && [ ! -z "$CURRENT_ITER" ]; then
    PROGRESS=$(echo "scale=2; $CURRENT_ITER * 100 / $TOTAL_ITERS" | bc)
    REMAINING=$((TOTAL_ITERS - CURRENT_ITER))
    echo "🎯 进度: $CURRENT_ITER / $TOTAL_ITERS ($PROGRESS%)"
    echo "⏳ 剩余步数: $REMAINING"
fi

echo ""
echo "=================================================="
echo "💡 监控命令："
echo "   实时查看: tail -f $LOG_FILE | grep 'Iter'"
echo "   查看进程: ps aux | grep mlx_lm | grep -v grep"
echo "   停止训练: kill $PID"
echo "=================================================="

