#!/usr/bin/env python3
"""
融合 V4.2 Adapter 到基础模型
生成可直接在 LM Studio 中使用的完整模型
"""

import os
import shutil
from pathlib import Path

# 配置路径
BASE_MODEL = "/Volumes/Pluto/Code/Model/lmstudio-models/local/XunDoc-30B-V4-Vision-Fused"
ADAPTER_PATH = "adapters_v42"
OUTPUT_PATH = "/Volumes/Pluto/Code/Model/lmstudio-models/local/XunDoc-30B-V42-Final"

def main():
    print("=" * 60)
    print("🔄 融合 V4.2 Adapter 到基础模型")
    print("=" * 60)
    
    print(f"\n📁 基础模型: {BASE_MODEL}")
    print(f"📁 Adapter: {ADAPTER_PATH}")
    print(f"📁 输出路径: {OUTPUT_PATH}")
    
    # 使用 mlx_lm.fuse 融合模型
    print("\n🔄 开始融合...")
    
    import subprocess
    result = subprocess.run([
        "python", "-m", "mlx_lm.fuse",
        "--model", BASE_MODEL,
        "--adapter-path", ADAPTER_PATH,
        "--save-path", OUTPUT_PATH,
        "--de-quantize"  # 保持原始精度
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ 融合完成！")
        print("=" * 60)
        print(f"\n📁 模型已保存到: {OUTPUT_PATH}")
        print("\n🎯 在 LM Studio 中使用:")
        print("   1. 打开 LM Studio")
        print("   2. 点击左侧 'My Models'")
        print("   3. 点击 'Add Model' -> 'Add local model'")
        print(f"   4. 选择目录: {OUTPUT_PATH}")
        print("   5. 加载并开始使用！")
    else:
        print(f"\n❌ 融合失败，错误码: {result.returncode}")

if __name__ == "__main__":
    main()


