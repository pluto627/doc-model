#!/usr/bin/env python3
"""
Web应用测试脚本
快速验证所有依赖是否正确安装
"""
import sys

def test_imports():
    """测试必要的库是否可以导入"""
    print("🧪 测试依赖库...")
    
    tests = {
        "Flask": lambda: __import__("flask"),
        "Pillow": lambda: __import__("PIL"),
        "Werkzeug": lambda: __import__("werkzeug"),
    }
    
    optional_tests = {
        "MLX": lambda: __import__("mlx"),
        "MLX-LM": lambda: __import__("mlx_lm"),
    }
    
    passed = 0
    failed = 0
    
    # 必需的库
    print("\n必需的库:")
    for name, test_func in tests.items():
        try:
            test_func()
            print(f"  ✅ {name}")
            passed += 1
        except ImportError as e:
            print(f"  ❌ {name} - 未安装")
            failed += 1
    
    # 可选的库
    print("\n可选的库（模型推理）:")
    for name, test_func in optional_tests.items():
        try:
            test_func()
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - 未安装（将使用模拟模式）")
    
    print(f"\n总结: {passed} 通过, {failed} 失败")
    
    if failed > 0:
        print("\n❌ 有必需的库未安装，请运行:")
        print("   pip install flask pillow werkzeug")
        return False
    else:
        print("\n✅ 所有必需的库都已安装!")
        return True


def test_directories():
    """测试目录结构"""
    import os
    
    print("\n🗂️  测试目录结构...")
    
    dirs = [
        "templates",
        "static",
        "uploads"
    ]
    
    for d in dirs:
        if os.path.exists(d):
            print(f"  ✅ {d}/")
        else:
            print(f"  ⚠️  {d}/ - 不存在，将在启动时创建")
    
    return True


def test_files():
    """测试必需的文件"""
    import os
    
    print("\n📄 测试必需文件...")
    
    files = {
        "app.py": "Web应用主程序",
        "templates/index.html": "前端页面",
        "config.py": "配置文件",
    }
    
    for f, desc in files.items():
        if os.path.exists(f):
            print(f"  ✅ {f} ({desc})")
        else:
            print(f"  ❌ {f} ({desc}) - 缺失!")
            return False
    
    return True


def test_config():
    """测试配置"""
    print("\n⚙️  测试配置...")
    
    try:
        import config
        print(f"  ✅ 配置文件加载成功")
        print(f"  📁 模型路径: {config.SOURCE_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"  ❌ 配置加载失败: {str(e)}")
        return False


def main():
    """主测试流程"""
    print("=" * 60)
    print("🏥 医疗图像分析Web应用 - 环境测试")
    print("=" * 60)
    
    all_passed = True
    
    # 运行所有测试
    all_passed &= test_imports()
    all_passed &= test_directories()
    all_passed &= test_files()
    all_passed &= test_config()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过! 可以启动Web应用了")
        print("\n运行以下命令启动服务器:")
        print("  ./start_server.sh")
        print("  或")
        print("  python3 app.py")
    else:
        print("❌ 有测试失败，请检查上述错误并修复")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

