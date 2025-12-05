#!/usr/bin/env python3
"""
数据预处理脚本
处理医疗视觉语言数据，准备训练格式
"""
import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    DataConfig, EMPATHY_TEMPLATES, 
    STYLE_GUIDELINES, PENALTY_WORDS, REWARD_WORDS,
    create_directories
)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

console = Console()


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.raw_dir = Path(self.config.raw_data_dir)
        self.processed_dir = Path(self.config.processed_data_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def process_image(self, image_path: str, output_path: str = None) -> Optional[str]:
        """
        处理单张图像
        - 调整大小
        - 标准化
        - 保存为统一格式
        """
        try:
            if not os.path.exists(image_path):
                return None
                
            img = Image.open(image_path)
            
            # 转换为RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 调整大小，保持宽高比
            max_size = self.config.image_size
            ratio = min(max_size / img.width, max_size / img.height)
            
            if ratio < 1:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # 保存
            if output_path:
                img.save(output_path, "JPEG", quality=95)
                return output_path
            
            return image_path
            
        except Exception as e:
            console.print(f"[yellow]⚠️ 图像处理失败 {image_path}: {str(e)}[/yellow]")
            return None
    
    def enhance_response_with_empathy(self, response: str) -> str:
        """
        增强回复的人情味
        """
        # 检查是否已经有人情味表达
        has_empathy = any(word in response for word in REWARD_WORDS[:3])
        
        if not has_empathy and len(response) > 50:
            # 随机添加人情味开头
            prefix = random.choice(EMPATHY_TEMPLATES)
            response = f"{prefix}\n\n{response}"
        
        # 添加结尾关怀
        if "?" not in response[-50:] and len(response) > 100:
            endings = [
                "\n\n如有其他疑问，请随时告诉我。",
                "\n\n希望这些信息对您有帮助。",
                "\n\n如果您还有其他问题，我很乐意为您解答。",
                "\n\n请放心，有任何问题都可以问我。"
            ]
            response += random.choice(endings)
        
        return response
    
    def check_response_quality(self, response: str) -> Tuple[float, List[str]]:
        """
        检查回复质量，返回惩罚分数和问题列表
        """
        penalty_score = 0.0
        issues = []
        
        # 检查惩罚词汇
        for word in PENALTY_WORDS:
            if word in response:
                penalty_score += 0.1
                issues.append(f"包含过于武断的表达: '{word}'")
        
        # 检查回复长度
        if len(response) < 50:
            penalty_score += 0.2
            issues.append("回复过短，缺乏详细解释")
        
        # 检查是否有专业内容
        medical_keywords = ["建议", "可能", "情况", "检查", "治疗", "症状", "医生"]
        has_medical_content = any(word in response for word in medical_keywords)
        if not has_medical_content:
            penalty_score += 0.1
            issues.append("缺乏专业医疗内容")
        
        # 奖励人情味表达
        reward_count = sum(1 for word in REWARD_WORDS if word in response)
        penalty_score -= reward_count * 0.05
        
        return max(0, penalty_score), issues
    
    def format_conversation_for_training(
        self, 
        conversations: List[Dict], 
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        将对话格式化为训练格式
        """
        # 构建Qwen3-VL格式
        messages = []
        
        for conv in conversations:
            role = conv.get("role", "user")
            content = conv.get("content", "")
            
            if role == "user":
                msg_content = []
                
                # 如果有图像，添加图像标记
                if image_path and messages == []:  # 只在第一个用户消息添加图像
                    msg_content.append({
                        "type": "image",
                        "image": image_path
                    })
                
                msg_content.append({
                    "type": "text",
                    "text": content
                })
                
                messages.append({
                    "role": "user",
                    "content": msg_content
                })
                
            elif role == "assistant":
                # 增强人情味
                enhanced_content = self.enhance_response_with_empathy(content)
                
                # 检查质量
                penalty, issues = self.check_response_quality(enhanced_content)
                
                messages.append({
                    "role": "assistant",
                    "content": enhanced_content
                })
        
        return {
            "messages": messages,
            "image_path": image_path
        }
    
    def process_dataset(self, dataset_name: str) -> List[Dict]:
        """
        处理单个数据集
        """
        console.print(f"\n[bold blue]📦 处理数据集: {dataset_name}[/bold blue]")
        
        data_path = self.raw_dir / dataset_name / "data.json"
        if not data_path.exists():
            console.print(f"[red]❌ 数据文件不存在: {data_path}[/red]")
            return []
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        processed_data = []
        image_output_dir = self.processed_dir / "images" / dataset_name
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(f"处理 {dataset_name}...", total=len(raw_data))
            
            for i, item in enumerate(raw_data):
                # 处理图像
                processed_image_path = None
                if item.get("image_path"):
                    output_img_path = str(image_output_dir / f"img_{i}.jpg")
                    processed_image_path = self.process_image(
                        item["image_path"], 
                        output_img_path
                    )
                
                # 处理对话
                if item.get("conversations"):
                    formatted = self.format_conversation_for_training(
                        item["conversations"],
                        processed_image_path
                    )
                    
                    processed_data.append({
                        "id": item.get("id", f"{dataset_name}_{i}"),
                        **formatted,
                        "metadata": item.get("metadata", {}),
                        "source": dataset_name
                    })
                
                progress.update(task, advance=1)
        
        console.print(f"[green]✅ 处理完成: {len(processed_data)} 条数据[/green]")
        return processed_data
    
    def create_train_val_split(
        self, 
        data: List[Dict], 
        val_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        创建训练/验证集划分
        """
        random.shuffle(data)
        split_idx = int(len(data) * (1 - val_ratio))
        return data[:split_idx], data[split_idx:]
    
    def save_processed_data(self, train_data: List[Dict], val_data: List[Dict]):
        """
        保存处理后的数据
        """
        # 保存训练数据
        train_file = self.processed_dir / "train.json"
        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        # 保存验证数据
        val_file = self.processed_dir / "val.json"
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        # 保存JSONL格式（用于训练）
        train_jsonl = self.processed_dir / "train.jsonl"
        with open(train_jsonl, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        val_jsonl = self.processed_dir / "val.jsonl"
        with open(val_jsonl, "w", encoding="utf-8") as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        console.print(f"[green]✅ 数据已保存到 {self.processed_dir}[/green]")
        console.print(f"   训练集: {len(train_data)} 条")
        console.print(f"   验证集: {len(val_data)} 条")
    
    def generate_penalty_reward_data(self) -> List[Dict]:
        """
        生成用于惩罚机制的对比数据
        展示好的回复和差的回复
        """
        console.print("\n[bold blue]📝 生成惩罚/奖励对比数据...[/bold blue]")
        
        contrast_pairs = [
            {
                "question": "我的血压有点高，怎么办？",
                "good_response": "我理解您的担忧，血压偏高确实需要关注。首先，请不要过度焦虑，因为轻度高血压是可以通过生活方式调整来改善的。建议您：1) 减少盐分摄入 2) 保持规律运动 3) 控制体重 4) 保证充足睡眠。如果持续偏高，建议咨询心内科医生。您有其他想了解的吗？",
                "bad_response": "血压高就是高血压，必须吃药，你去医院看看吧。"
            },
            {
                "question": "检查报告显示有结节，是不是癌症？",
                "good_response": "我能深深理解您此刻的担忧，发现结节确实会让人感到恐惧。但请让我帮您理性地分析一下：首先，大多数结节都是良性的，恶性的概率其实很低。关键是要看结节的大小、形态、边缘特征等。建议您：1) 不要过度恐慌 2) 咨询专科医生的意见 3) 如果医生建议随访观察，请按时复查。有什么具体问题我可以帮您解答？",
                "bad_response": "肯定是要去检查的，这种事情谁说得准呢，你自己想办法吧。"
            },
            {
                "question": "我经常头痛，需要做什么检查？",
                "good_response": "感谢您向我咨询。头痛是很常见的症状，我理解这给您带来的不适和困扰。头痛的原因有很多，可能与睡眠、压力、颈椎问题等有关。为了更好地了解原因，建议您记录一下：1) 头痛的部位和性质 2) 发作频率和持续时间 3) 是否有伴随症状。如果头痛频繁或剧烈，可以考虑做头部CT或MRI检查。您方便描述一下头痛的具体情况吗？",
                "bad_response": "头痛没什么大不了的，吃点止痛药就行了。"
            }
        ]
        
        data_list = []
        for i, pair in enumerate(contrast_pairs):
            # 好的回复
            data_list.append({
                "id": f"contrast_good_{i}",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": pair["question"]}]},
                    {"role": "assistant", "content": pair["good_response"]}
                ],
                "image_path": None,
                "reward_label": 1.0,  # 奖励
                "source": "contrast_positive"
            })
            
            # 差的回复（用于对比学习）
            data_list.append({
                "id": f"contrast_bad_{i}",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": pair["question"]}]},
                    {"role": "assistant", "content": pair["bad_response"]}
                ],
                "image_path": None,
                "reward_label": -1.0,  # 惩罚
                "source": "contrast_negative"
            })
        
        console.print(f"[green]✅ 生成 {len(data_list)} 条对比数据[/green]")
        return data_list


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold green]🔧 医疗数据预处理[/bold green]\n"
        "处理下载的数据集，准备训练格式",
        border_style="green"
    ))
    
    # 创建目录
    create_directories()
    
    # 初始化预处理器
    preprocessor = DataPreprocessor()
    
    all_data = []
    
    # 处理各个数据集
    datasets_to_process = [
        "medical_vision_llm",
        "aquiles_medical_vision", 
        "medtrinity",
        "empathy_data"
    ]
    
    for dataset_name in datasets_to_process:
        data = preprocessor.process_dataset(dataset_name)
        all_data.extend(data)
    
    # 生成对比数据
    contrast_data = preprocessor.generate_penalty_reward_data()
    all_data.extend(contrast_data)
    
    # 生成药物知识库训练数据
    console.print("\n[bold blue]💊 生成药物知识库训练数据...[/bold blue]")
    try:
        from generate_drug_training_data import DrugTrainingDataGenerator
        drug_generator = DrugTrainingDataGenerator()
        drug_data = drug_generator.generate_all_training_data()
        all_data.extend(drug_data)
        console.print(f"[green]✅ 添加药物训练数据: {len(drug_data)} 条[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️  药物数据生成失败: {str(e)}[/yellow]")
    
    # 划分训练/验证集
    console.print("\n[bold blue]📊 划分训练/验证集...[/bold blue]")
    train_data, val_data = preprocessor.create_train_val_split(all_data, val_ratio=0.1)
    
    # 保存数据
    preprocessor.save_processed_data(train_data, val_data)
    
    # 总结
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        f"[bold green]📊 预处理完成总结[/bold green]\n"
        f"总数据量: {len(all_data)} 条\n"
        f"训练集: {len(train_data)} 条\n"
        f"验证集: {len(val_data)} 条\n"
        f"输出目录: {PROCESSED_DATA_DIR}",
        border_style="green"
    ))


if __name__ == "__main__":
    main()

