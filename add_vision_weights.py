#!/usr/bin/env python3
"""
添加视觉权重到 XunDoc-30B-V3-Precision 模型
从 Qwen/Qwen3-VL-30B-A3B-Instruct 下载视觉权重并合并
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import mlx.core as mx
from rich.console import Console

console = Console()

# 配置路径
SOURCE_REPO = "Qwen/Qwen3-VL-30B-A3B-Instruct"
TARGET_MODEL_DIR = Path("/Volumes/Pluto/Code/Model/lmstudio-models/local/XunDoc-30B-V3-Precision")
OUTPUT_MODEL_DIR = Path("/Volumes/Pluto/Code/Model/lmstudio-models/local/XunDoc-30B-V3-Precision-Vision")


def map_weight_name(name: str) -> str:
    """将 HuggingFace 格式的权重名映射到 MLX 格式"""
    if name.startswith("model.visual."):
        return name.replace("model.visual.", "vision_tower.")
    return name


def download_vision_weights():
    """下载视觉权重文件"""
    console.print("[cyan]📥 下载视觉权重文件...[/cyan]")
    
    vision_file = hf_hub_download(
        SOURCE_REPO, 
        "model-00013-of-00013.safetensors",
        local_dir="./temp_vision_weights"
    )
    
    config_file = hf_hub_download(
        SOURCE_REPO,
        "config.json",
        local_dir="./temp_vision_weights"
    )
    
    preprocessor_file = hf_hub_download(
        SOURCE_REPO,
        "preprocessor_config.json", 
        local_dir="./temp_vision_weights"
    )
    
    video_preprocessor_file = hf_hub_download(
        SOURCE_REPO,
        "video_preprocessor_config.json",
        local_dir="./temp_vision_weights"
    )
    
    console.print("[green]✅ 视觉权重下载完成[/green]")
    return vision_file, config_file, preprocessor_file, video_preprocessor_file


def extract_vision_weights(vision_file: str) -> dict:
    """从下载的文件中提取视觉权重"""
    console.print("[cyan]🔧 提取视觉权重...[/cyan]")
    
    # 使用 MLX 原生加载 safetensors (支持 bfloat16)
    all_weights = mx.load(vision_file)
    
    vision_weights = {}
    for key, tensor in all_weights.items():
        if "visual" in key.lower():
            new_key = map_weight_name(key)
            vision_weights[new_key] = tensor
                
    console.print(f"[green]✅ 提取了 {len(vision_weights)} 个视觉权重[/green]")
    return vision_weights


def update_model_config(config_file: str, target_dir: Path, output_dir: Path):
    """更新模型配置，添加 vision_config"""
    console.print("[cyan]📝 更新模型配置...[/cyan]")
    
    with open(config_file) as f:
        source_config = json.load(f)
    
    target_config_path = target_dir / "config.json"
    with open(target_config_path) as f:
        target_config = json.load(f)
    
    if "vision_config" in source_config:
        target_config["vision_config"] = source_config["vision_config"]
        console.print("[green]  ✅ 添加 vision_config[/green]")
    
    vision_fields = [
        "image_token_id", "video_token_id", 
        "vision_start_token_id", "vision_end_token_id"
    ]
    for field in vision_fields:
        if field in source_config:
            target_config[field] = source_config[field]
    
    output_config_path = output_dir / "config.json"
    with open(output_config_path, "w") as f:
        json.dump(target_config, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✅ 配置已保存[/green]")
    return target_config


def merge_weights(target_dir: Path, output_dir: Path, vision_weights: dict):
    """合并视觉权重到模型"""
    console.print("[cyan]🔗 合并视觉权重到模型...[/cyan]")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制原始模型文件
    for f in target_dir.iterdir():
        if f.is_file() and f.name != "config.json":
            shutil.copy2(f, output_dir / f.name)
            console.print(f"  📄 复制 {f.name}")
    
    # 读取原始 index 文件
    index_path = target_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index_data = json.load(f)
    
    # 保存视觉权重
    vision_file_name = "model-vision.safetensors"
    vision_file_path = output_dir / vision_file_name
    
    console.print(f"[cyan]💾 保存视觉权重到 {vision_file_name}...[/cyan]")
    mx.save_safetensors(str(vision_file_path), vision_weights)
    
    # 更新 index 文件
    weight_map = index_data.get("weight_map", {})
    for key in vision_weights.keys():
        weight_map[key] = vision_file_name
    
    index_data["weight_map"] = weight_map
    
    output_index_path = output_dir / "model.safetensors.index.json"
    with open(output_index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    
    console.print(f"[green]✅ 权重合并完成[/green]")
    console.print(f"  📊 总权重数: {len(weight_map)}")
    console.print(f"  👁️ 视觉权重数: {len(vision_weights)}")


def copy_preprocessor_configs(preprocessor_file: str, video_preprocessor_file: str, output_dir: Path):
    """复制预处理器配置文件"""
    console.print("[cyan]📋 复制预处理器配置...[/cyan]")
    
    shutil.copy2(preprocessor_file, output_dir / "preprocessor_config.json")
    shutil.copy2(video_preprocessor_file, output_dir / "video_preprocessor_config.json")
    
    console.print("[green]✅ 预处理器配置已复制[/green]")


def verify_model(output_dir: Path):
    """验证合并后的模型"""
    console.print("\n[cyan]🔍 验证模型...[/cyan]")
    
    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "model-vision.safetensors",
        "preprocessor_config.json",
        "video_preprocessor_config.json"
    ]
    
    all_present = True
    for f in required_files:
        path = output_dir / f
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            console.print(f"  ✅ {f} ({size:.2f} MB)")
        else:
            console.print(f"  ❌ {f} 缺失")
            all_present = False
    
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index_data = json.load(f)
    
    weight_map = index_data.get("weight_map", {})
    vision_keys = [k for k in weight_map.keys() if "vision" in k.lower()]
    
    console.print(f"\n  📊 权重统计:")
    console.print(f"     总权重: {len(weight_map)}")
    console.print(f"     视觉权重: {len(vision_keys)}")
    
    if len(vision_keys) > 0 and all_present:
        console.print("\n[bold green]🎉 模型验证通过！视觉权重已成功添加[/bold green]")
        return True
    else:
        console.print("\n[bold red]❌ 模型验证失败[/bold red]")
        return False


def main():
    console.print("\n" + "="*60)
    console.print("[bold cyan]🔧 添加视觉权重到 XunDoc-30B-V3-Precision[/bold cyan]")
    console.print("="*60 + "\n")
    
    console.print(f"[blue]源模型: {SOURCE_REPO}[/blue]")
    console.print(f"[blue]目标模型: {TARGET_MODEL_DIR}[/blue]")
    console.print(f"[blue]输出目录: {OUTPUT_MODEL_DIR}[/blue]\n")
    
    if not TARGET_MODEL_DIR.exists():
        console.print(f"[red]❌ 目标模型不存在: {TARGET_MODEL_DIR}[/red]")
        return
    
    try:
        # 1. 下载视觉权重
        vision_file, config_file, preprocessor_file, video_preprocessor_file = download_vision_weights()
        
        # 2. 提取视觉权重
        vision_weights = extract_vision_weights(vision_file)
        
        # 3. 创建输出目录并合并权重
        merge_weights(TARGET_MODEL_DIR, OUTPUT_MODEL_DIR, vision_weights)
        
        # 4. 更新配置
        update_model_config(config_file, TARGET_MODEL_DIR, OUTPUT_MODEL_DIR)
        
        # 5. 复制预处理器配置
        copy_preprocessor_configs(preprocessor_file, video_preprocessor_file, OUTPUT_MODEL_DIR)
        
        # 6. 验证
        success = verify_model(OUTPUT_MODEL_DIR)
        
        if success:
            console.print("\n" + "="*60)
            console.print("[bold green]✅ 视觉权重添加完成！[/bold green]")
            console.print(f"\n新模型位置: {OUTPUT_MODEL_DIR}")
            console.print("\n[yellow]推荐训练步数: 1000 步[/yellow]")
            console.print("[dim]（用于视觉-语言对齐微调）[/dim]")
            console.print("="*60)
        
        # 清理临时文件
        temp_dir = Path("./temp_vision_weights")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            console.print("\n[dim]临时文件已清理[/dim]")
            
    except Exception as e:
        console.print(f"\n[red]❌ 错误: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
