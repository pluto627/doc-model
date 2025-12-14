#!/usr/bin/env python3
"""
测试 XunDoc-30B-V4 视觉功能
"""
import sys
from pathlib import Path
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_PATH = "/Volumes/Pluto/Code/Model/lmstudio-models/local/XunDoc-30B-V4-Vision-Fused"

def main():
    print("=" * 60)
    print("🏥 XunDoc-30B-V4 视觉医疗模型测试")
    print("=" * 60)
    
    # 检查是否提供了图像路径
    if len(sys.argv) < 2:
        print("\n用法: python test_vision.py <图像路径> [问题]")
        print("\n示例:")
        print("  python test_vision.py image.jpg")
        print("  python test_vision.py xray.png '请分析这张X光片'")
        print("\n支持的图像格式: jpg, png, webp")
        return
    
    image_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "请描述这张图片中的内容，如果是医学影像请给出分析。"
    
    # 检查图像文件
    if not Path(image_path).exists():
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    print(f"\n📷 图像: {image_path}")
    print(f"❓ 问题: {question}")
    print("\n⏳ 加载模型...")
    
    try:
        # 加载模型
        model, processor = load(MODEL_PATH)
        config = load_config(MODEL_PATH)
        
        print("✅ 模型加载成功")
        print("\n🤖 生成回答中...\n")
        print("-" * 40)
        
        # 构建消息
        messages = [
            {"role": "user", "content": question}
        ]
        
        # 应用聊天模板
        prompt = apply_chat_template(processor, config, messages, add_generation_prompt=True)
        
        # 生成回答
        output = generate(
            model,
            processor,
            prompt,
            image=image_path,
            max_tokens=1024,
            verbose=False
        )
        
        print(output)
        print("-" * 40)
        print("\n✅ 完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

