#!/usr/bin/env python3
"""
测试训练好的V2模型
可以在命令行直接测试，也可以在LM Studio中使用
"""
import sys
from pathlib import Path

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️  mlx_lm未安装，请使用LM Studio测试")

def test_model_cli():
    """命令行测试模型"""
    if not MLX_AVAILABLE:
        print("\n请使用以下方式测试模型：")
        print("1. 打开LM Studio")
        print("2. 加载模型: Qwen3-VL-30B-Medical-V2-Accuracy")
        print("3. 开始聊天测试\n")
        return
    
    print("=" * 60)
    print("🎯 Qwen3-VL-30B 医疗模型 V2 测试")
    print("=" * 60)
    print()
    
    # 模型路径
    base_model = "/Users/plutoguo/.lmstudio/models/lmstudio-community/Qwen3-VL-30B-Medical-Finetuned"
    adapter_path = "/Users/plutoguo/Desktop/training/adapters_v2"
    
    print("📦 加载模型...")
    print(f"   基础模型: {Path(base_model).name}")
    print(f"   Adapter: {Path(adapter_path).name}")
    print()
    
    try:
        model, tokenizer = load(base_model, adapter_path=adapter_path)
        print("✅ 模型加载成功！")
        print()
        
        # 测试案例
        test_cases = [
            "我的血压140/90，需要担心吗？",
            "感冒发烧38.5度，该怎么办？",
            "最近总是感到疲劳，可能是什么原因？"
        ]
        
        print("📝 开始测试...")
        print("=" * 60)
        print()
        
        for i, question in enumerate(test_cases, 1):
            print(f"测试 {i}/{len(test_cases)}")
            print(f"问题: {question}")
            print()
            
            # 生成回复
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            
            response = generate(
                model, 
                tokenizer,
                prompt=prompt,
                max_tokens=500,
                temp=0.7,
                verbose=False
            )
            
            print(f"回答: {response}")
            print()
            print("-" * 60)
            print()
        
        print("=" * 60)
        print("✅ 测试完成！")
        print()
        print("💡 评估标准：")
        print("   1. 是否有同理心表达（人情味）")
        print("   2. 是否提供准确的医学信息（精度）")
        print("   3. 是否避免武断表达")
        print("   4. 是否给出具体建议")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        print()
        print("请尝试在LM Studio中测试")


def print_lmstudio_guide():
    """打印LM Studio使用指南"""
    print()
    print("=" * 60)
    print("📖 LM Studio 使用指南")
    print("=" * 60)
    print()
    print("1️⃣  打开LM Studio应用")
    print()
    print("2️⃣  在侧边栏找到模型列表")
    print("   - 点击 'My Models' 或 '我的模型'")
    print()
    print("3️⃣  找到新训练的模型")
    print("   模型名称: Qwen3-VL-30B-Medical-V2-Accuracy")
    print("   位置: lmstudio-community/")
    print()
    print("4️⃣  加载模型")
    print("   - 点击模型卡片")
    print("   - 点击 'Load Model' 或 '加载模型'")
    print("   - 等待模型加载完成（可能需要1-2分钟）")
    print()
    print("5️⃣  开始聊天测试")
    print("   - 切换到 'Chat' 或 '聊天' 标签")
    print("   - 输入测试问题")
    print()
    print("📝 推荐测试问题：")
    print("   • 我的血压140/90，需要担心吗？")
    print("   • 感冒发烧38.5度，该怎么办？")
    print("   • 最近总是感到疲劳，可能是什么原因？")
    print("   • （可以上传医学图像进行分析）")
    print()
    print("✅ 期待看到的改进：")
    print("   1. 更准确的医学分析")
    print("   2. 保持温暖的表达")
    print("   3. 更科学的建议")
    print("   4. 减少武断的判断")
    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    print()
    print("🏥 医疗VLM V2模型测试工具")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        test_model_cli()
    else:
        print_lmstudio_guide()
        print()
        print("💡 提示：")
        print("   - 运行 'python test_v2_model.py --cli' 进行命令行测试")
        print("   - 或直接在LM Studio中使用（推荐）")
        print()

