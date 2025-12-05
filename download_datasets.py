#!/usr/bin/env python3
"""
医疗数据集下载脚本
从 HuggingFace 下载多个医疗视觉语言数据集
"""
import os
import sys
import json
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import requests

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RAW_DATA_DIR, DATASETS, create_directories
)

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  huggingface_hub 未安装，部分功能可能受限")

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

console = Console()


class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, output_dir: str = RAW_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_medical_vision_llm(self, max_samples: int = 5000) -> str:
        """
        下载 robailleo/medical-vision-llm-dataset
        医疗视觉问答数据集
        """
        console.print("\n[bold blue]📥 下载 Medical Vision LLM Dataset...[/bold blue]")
        
        output_path = self.output_dir / "medical_vision_llm"
        output_path.mkdir(exist_ok=True)
        
        try:
            dataset = load_dataset(
                "robailleo/medical-vision-llm-dataset",
                split="train",
                streaming=True
            )
            
            data_list = []
            image_dir = output_path / "images"
            image_dir.mkdir(exist_ok=True)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("下载数据...", total=max_samples)
                
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break
                        
                    sample = {
                        "id": f"mvl_{i}",
                        "conversations": [],
                        "image_path": None
                    }
                    
                    # 处理图像
                    if "image" in item and item["image"] is not None:
                        try:
                            img = item["image"]
                            img_path = image_dir / f"img_{i}.jpg"
                            if hasattr(img, 'save'):
                                img.save(str(img_path), "JPEG")
                                sample["image_path"] = str(img_path)
                        except Exception as e:
                            pass
                    
                    # 处理对话
                    if "question" in item and "answer" in item:
                        sample["conversations"] = [
                            {"role": "user", "content": item.get("question", "")},
                            {"role": "assistant", "content": item.get("answer", "")}
                        ]
                    elif "conversations" in item:
                        sample["conversations"] = item["conversations"]
                    elif "text" in item:
                        # 尝试从text字段解析
                        sample["conversations"] = [
                            {"role": "user", "content": "请分析这张医学图像"},
                            {"role": "assistant", "content": item.get("text", "")}
                        ]
                    
                    if sample["conversations"]:
                        data_list.append(sample)
                    
                    progress.update(task, advance=1)
            
            # 保存数据
            output_file = output_path / "data.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            console.print(f"[green]✅ 已下载 {len(data_list)} 条数据到 {output_path}[/green]")
            return str(output_path)
            
        except Exception as e:
            console.print(f"[red]❌ 下载失败: {str(e)}[/red]")
            return ""

    def download_aquiles_medical_vision(self, max_samples: int = 5000) -> str:
        """
        下载 Aquiles-ai/Medical-Vision
        医疗影像数据集
        """
        console.print("\n[bold blue]📥 下载 Aquiles Medical Vision Dataset...[/bold blue]")
        
        output_path = self.output_dir / "aquiles_medical_vision"
        output_path.mkdir(exist_ok=True)
        
        try:
            dataset = load_dataset(
                "Aquiles-ai/Medical-Vision",
                split="train",
                streaming=True
            )
            
            data_list = []
            image_dir = output_path / "images"
            image_dir.mkdir(exist_ok=True)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("下载数据...", total=max_samples)
                
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break
                    
                    sample = {
                        "id": f"amv_{i}",
                        "conversations": [],
                        "image_path": None,
                        "modality": item.get("modality", "unknown"),
                        "body_part": item.get("body_part", "unknown")
                    }
                    
                    # 处理图像
                    if "image" in item and item["image"] is not None:
                        try:
                            img = item["image"]
                            img_path = image_dir / f"img_{i}.jpg"
                            if hasattr(img, 'save'):
                                img.save(str(img_path), "JPEG")
                                sample["image_path"] = str(img_path)
                        except Exception as e:
                            pass
                    
                    # 构建对话
                    question = item.get("question", f"请分析这张{sample.get('modality', '医学')}图像")
                    answer = item.get("answer", item.get("caption", item.get("description", "")))
                    
                    if answer:
                        sample["conversations"] = [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                        data_list.append(sample)
                    
                    progress.update(task, advance=1)
            
            # 保存数据
            output_file = output_path / "data.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            console.print(f"[green]✅ 已下载 {len(data_list)} 条数据到 {output_path}[/green]")
            return str(output_path)
            
        except Exception as e:
            console.print(f"[red]❌ 下载失败: {str(e)}[/red]")
            return ""

    def download_medtrinity(self, max_samples: int = 10000) -> str:
        """
        下载 UCSC-VLAA/MedTrinity-25M
        大规模医疗三元组数据集
        """
        console.print("\n[bold blue]📥 下载 MedTrinity-25M Dataset...[/bold blue]")
        
        output_path = self.output_dir / "medtrinity"
        output_path.mkdir(exist_ok=True)
        
        try:
            # 使用demo子集，更快下载
            dataset = load_dataset(
                "UCSC-VLAA/MedTrinity-25M",
                "25M_demo",
                split="train",
                streaming=True
            )
            
            data_list = []
            image_dir = output_path / "images"
            image_dir.mkdir(exist_ok=True)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("下载数据...", total=max_samples)
                
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break
                    
                    sample = {
                        "id": f"mt_{i}",
                        "conversations": [],
                        "image_path": None,
                        "metadata": {}
                    }
                    
                    # 处理图像
                    if "image" in item and item["image"] is not None:
                        try:
                            img = item["image"]
                            img_path = image_dir / f"img_{i}.jpg"
                            if hasattr(img, 'save'):
                                img.save(str(img_path), "JPEG")
                                sample["image_path"] = str(img_path)
                        except Exception as e:
                            pass
                    
                    # 提取元数据
                    for key in ["modality", "organ", "abnormality", "caption"]:
                        if key in item:
                            sample["metadata"][key] = item[key]
                    
                    # 构建对话
                    caption = item.get("caption", "")
                    if caption:
                        modality = item.get("modality", "医学")
                        organ = item.get("organ", "")
                        
                        question = f"请分析这张{modality}图像"
                        if organ:
                            question += f"，关注{organ}部位"
                        
                        # 添加人情味回复
                        empathy_prefix = "我来帮您分析这张图像。"
                        enhanced_answer = f"{empathy_prefix}{caption}"
                        
                        sample["conversations"] = [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": enhanced_answer}
                        ]
                        data_list.append(sample)
                    
                    progress.update(task, advance=1)
            
            # 保存数据
            output_file = output_path / "data.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            console.print(f"[green]✅ 已下载 {len(data_list)} 条数据到 {output_path}[/green]")
            return str(output_path)
            
        except Exception as e:
            console.print(f"[red]❌ 下载失败: {str(e)}[/red]")
            return ""

    def create_synthetic_empathy_data(self, num_samples: int = 2000) -> str:
        """
        创建合成的人情味训练数据
        用于增强模型的同理心表达能力
        """
        console.print("\n[bold blue]📝 生成人情味训练数据...[/bold blue]")
        
        output_path = self.output_dir / "empathy_data"
        output_path.mkdir(exist_ok=True)
        
        # 医疗场景问答模板
        templates = [
            {
                "user": "我的血压有点高，是150/95，我很担心。",
                "assistant": "我理解您的担忧，血压偏高确实需要关注。150/95 mmHg属于轻度高血压范围。不过请不要过于焦虑，这个数值是可以通过生活方式调整来改善的。建议您：1) 减少盐分摄入 2) 适当运动 3) 保持规律作息 4) 定期监测血压。如果持续偏高，建议咨询心内科医生进行专业评估。您有其他想了解的吗？"
            },
            {
                "user": "我的血糖检测结果是7.2，这正常吗？",
                "assistant": "感谢您分享检测结果。空腹血糖7.2 mmol/L略高于正常范围（正常值一般在3.9-6.1 mmol/L）。我能感受到您对此的关切，这是很正常的反应。这个数值提示需要进一步关注，但不必过度担心。建议您：1) 注意饮食，减少精制碳水化合物 2) 增加运动量 3) 一周后复查空腹血糖。如果您方便的话，可以做一个糖化血红蛋白检测，这能更全面地反映近3个月的血糖情况。有任何疑问都可以问我。"
            },
            {
                "user": "体检报告显示我有脂肪肝，严重吗？",
                "assistant": "我理解收到这样的结果可能让您有些担心。脂肪肝是目前很常见的情况，轻度脂肪肝通常是可逆的，通过生活方式调整可以改善甚至恢复正常。重要的是了解脂肪肝的程度：轻度、中度还是重度。轻度脂肪肝一般不会对健康造成严重影响，但需要积极干预。建议您：1) 控制体重 2) 减少高脂高糖饮食 3) 戒酒或限酒 4) 规律运动。如果是中重度，建议定期复查肝功能。您对饮食调整有什么疑问吗？"
            },
            {
                "user": "我最近胸口有时候会闷，是不是心脏有问题？",
                "assistant": "我非常理解您的担忧，胸闷确实会让人感到不安。首先请放心，胸闷不一定是心脏问题，还可能与呼吸系统、消化系统、焦虑情绪等有关。但为了安全起见，我建议您：1) 注意胸闷发作的时间、持续时长、是否与运动或情绪有关 2) 如果伴有胸痛、气短、出汗等症状，请尽快就医 3) 可以先做一个心电图检查排除心脏问题。请问您的胸闷是持续性的还是间歇性的？有没有其他伴随症状？"
            },
            {
                "user": "我孩子的血常规显示白细胞偏高，是不是很严重？",
                "assistant": "作为家长，您一定很担心孩子的健康，这种心情我完全理解。白细胞偏高最常见的原因是身体正在对抗感染，这其实是免疫系统正常工作的表现。孩子最近有没有感冒、发烧或其他不适症状呢？如果是普通感染引起的白细胞升高，随着感染好转会自然恢复正常。建议您：1) 观察孩子是否有发热、咳嗽等症状 2) 保证充足休息和水分摄入 3) 1-2周后复查血常规。如果白细胞持续升高或孩子有其他异常，请带孩子去儿科就诊。有什么我可以帮您详细解释的吗？"
            },
            {
                "user": "我的尿酸值有点高，会不会得痛风？",
                "assistant": "感谢您关注自己的健康。尿酸偏高确实是痛风的风险因素之一，但并不是所有高尿酸的人都会发展成痛风。大约只有10-15%的高尿酸血症患者会最终出现痛风症状。我理解您对此的担忧，但好消息是，通过积极的生活方式调整，很多人可以将尿酸控制在正常范围。建议您：1) 多喝水，每天2000ml以上 2) 减少高嘌呤食物（如动物内脏、海鲜、啤酒）3) 适当运动但避免剧烈运动 4) 保持健康体重。定期复查尿酸水平也很重要。您平时的饮食习惯是怎样的呢？"
            },
            {
                "user": "CT报告说肺部有小结节，我特别害怕是不是肺癌。",
                "assistant": "我能深深感受到您此刻的焦虑和担忧。发现肺结节确实会让人感到恐惧，但请让我帮您理性地分析一下。首先，大多数肺部小结节（特别是直径小于6mm）是良性的，可能是之前的感染留下的痕迹、淋巴结或其他良性病变。恶性的概率其实很低。关键是要看结节的大小、形态、边缘特征等。建议您：1) 不要过度恐慌，等待专业医生的评估 2) 如果结节较小，可能只需定期随访观察 3) 咨询胸外科或呼吸科专家的意见 4) 保持良好心态，焦虑不利于健康。请问您的结节具体多大？报告有描述其他特征吗？"
            },
            {
                "user": "我经常失眠，安眠药吃多了对身体不好吧？",
                "assistant": "您提出了一个非常好的问题，这显示您对健康问题的认真态度。长期依赖安眠药确实不是最佳选择，您的担心是有道理的。失眠是一个需要综合管理的问题。我理解长期失眠给您带来的困扰，这种痛苦是真实的。建议您考虑以下方法：1) 建立规律的作息时间 2) 睡前避免使用电子设备 3) 尝试放松技巧如深呼吸、冥想 4) 保持适量运动但避免睡前剧烈运动 5) 限制咖啡因摄入。如果非药物方法效果不好，建议咨询睡眠专科医生，可能有更适合您的治疗方案。您的失眠大概持续多久了呢？"
            }
        ]
        
        # 扩展模板生成更多数据
        data_list = []
        variations = [
            ("我很担心", "我有点紧张", "我很焦虑", "我比较担忧"),
            ("是不是", "会不会", "是否", "可能是"),
            ("严重吗", "要紧吗", "有问题吗", "正常吗"),
        ]
        
        sample_id = 0
        for template in templates:
            # 原始模板
            data_list.append({
                "id": f"emp_{sample_id}",
                "conversations": [
                    {"role": "user", "content": template["user"]},
                    {"role": "assistant", "content": template["assistant"]}
                ],
                "image_path": None
            })
            sample_id += 1
            
            # 生成变体
            for _ in range(int(num_samples / len(templates) / 2)):
                modified_user = template["user"]
                for var_group in variations:
                    if var_group[0] in modified_user:
                        import random
                        modified_user = modified_user.replace(
                            var_group[0], 
                            random.choice(var_group[1:])
                        )
                
                data_list.append({
                    "id": f"emp_{sample_id}",
                    "conversations": [
                        {"role": "user", "content": modified_user},
                        {"role": "assistant", "content": template["assistant"]}
                    ],
                    "image_path": None
                })
                sample_id += 1
        
        # 保存数据
        output_file = output_path / "data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]✅ 已生成 {len(data_list)} 条人情味训练数据[/green]")
        return str(output_path)


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold green]🏥 医疗视觉语言模型数据下载器[/bold green]\n"
        "准备下载多个医疗数据集用于模型微调训练",
        border_style="green"
    ))
    
    # 创建目录
    create_directories()
    
    # 初始化下载器
    downloader = DatasetDownloader()
    
    downloaded_paths = []
    
    # 1. 下载 Medical Vision LLM Dataset
    path = downloader.download_medical_vision_llm(max_samples=5000)
    if path:
        downloaded_paths.append(path)
    
    # 2. 下载 Aquiles Medical Vision
    path = downloader.download_aquiles_medical_vision(max_samples=5000)
    if path:
        downloaded_paths.append(path)
    
    # 3. 下载 MedTrinity-25M
    path = downloader.download_medtrinity(max_samples=10000)
    if path:
        downloaded_paths.append(path)
    
    # 4. 生成人情味训练数据
    path = downloader.create_synthetic_empathy_data(num_samples=2000)
    if path:
        downloaded_paths.append(path)
    
    # 总结
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        f"[bold green]📊 下载完成总结[/bold green]\n"
        f"成功下载 {len(downloaded_paths)} 个数据集\n"
        f"数据存储于: {RAW_DATA_DIR}",
        border_style="green"
    ))
    
    for path in downloaded_paths:
        console.print(f"  ✅ {path}")
    
    return downloaded_paths


if __name__ == "__main__":
    main()

