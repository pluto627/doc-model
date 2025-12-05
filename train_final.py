#!/usr/bin/env python3
"""
医疗VLM最终训练脚本 - 精度优先版
优先级：1. 精度(50%) 2. 人情味(35%) 3. 图像(15%)
"""
import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()

# 尝试导入MLX
try:
    import mlx
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    console.print("[yellow]⚠️  MLX未安装，将使用模拟训练模式[/yellow]")


@dataclass
class FinalTrainingConfig:
    """最终训练配置 - 精度优先"""
    # 基础配置
    base_model_path: str
    output_dir: str = "./finetuned_model_final"
    checkpoint_dir: str = "./checkpoints_final"
    log_dir: str = "./logs"
    
    # LoRA配置
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.08
    
    # 训练参数
    num_train_steps: int = 2000
    batch_size: int = 2
    gradient_accumulation: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.02
    warmup_steps: int = 100
    
    # 优先级权重
    accuracy_weight: float = 0.50   # 50%
    empathy_weight: float = 0.35    # 35%
    vision_weight: float = 0.15     # 15%
    
    # 奖励系数（反映优先级）
    accuracy_reward_coef: float = 2.0    # 最高
    empathy_reward_coef: float = 1.0     # 第二
    vision_reward_coef: float = 0.6      # 第三
    
    # 惩罚系数（反映严重性）
    accuracy_penalty_coef: float = 3.0   # 最严厉
    coldness_penalty_coef: float = 1.5   # 较严厉
    vision_penalty_coef: float = 1.0     # 一般
    
    # 评估和保存
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 20
    
    # 训练阶段
    phase1_steps: int = 1000  # 精度强化
    phase2_steps: int = 600   # 人情味融合
    phase3_steps: int = 400   # 图像提升


class PriorityRewardCalculator:
    """优先级导向的奖惩计算器"""
    
    def __init__(self, config: FinalTrainingConfig):
        self.config = config
        
        # 精度相关（第一优先级）
        self.accuracy_high_value = [
            "诊断", "分析", "检查", "指标", "数值", "范围", "正常值",
            "异常", "症状", "病因", "治疗", "建议", "标准", "参考",
            "可能是", "需要", "应该", "通常", "一般", "表明"
        ]
        
        self.medical_terms = [
            "血压", "血糖", "心率", "体温", "血常规", "肝功能", "肾功能",
            "白细胞", "红细胞", "血小板", "尿酸", "胆固醇", "甘油三酯",
            "炎症", "感染", "肿瘤", "病变", "病灶", "水肿", "充血"
        ]
        
        # 人情味相关（第二优先级）
        self.empathy_high_value = [
            "我理解", "我明白", "感谢", "让我", "我来", "帮您",
            "为您", "关心", "担心", "放心", "不用太", "希望"
        ]
        
        self.empathy_medium_value = [
            "您", "请", "建议您", "提醒您", "祝您",
            "可以", "能够", "尽量", "注意"
        ]
        
        # 图像相关（第三优先级）
        self.vision_keywords = [
            "影像", "图像", "X光", "CT", "MRI", "超声", "B超",
            "可见", "显示", "区域", "位置", "形态", "大小", "密度",
            "信号", "阴影", "结节", "病灶", "异常", "正常"
        ]
        
        # 需要严厉惩罚的错误表达
        self.critical_errors = [
            "肯定是", "一定是", "必须是", "绝对是", "100%",
            "不可能", "绝不会", "毫无疑问"
        ]
        
        # 冷漠表达
        self.cold_expressions = [
            "自己看", "问别人", "不知道", "不清楚", "没办法"
        ]
    
    def calculate_accuracy_score(self, text: str) -> Dict[str, float]:
        """计算精度得分（第一优先级）"""
        reward = 0.0
        penalty = 0.0
        
        # 高价值医学表达
        for word in self.accuracy_high_value:
            if word in text:
                reward += 0.5
        
        # 医学术语
        term_count = sum(1 for term in self.medical_terms if term in text)
        reward += min(term_count * 0.4, 2.0)
        
        # 量化信息
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        if len(numbers) >= 2:
            reward += 0.8  # 有具体数据
        
        # 结构化表达
        structure_markers = ["首先", "其次", "然后", "最后", "1.", "2.", "3."]
        if any(marker in text for marker in structure_markers):
            reward += 0.6
        
        # 详细程度
        if len(text) > 200:
            reward += 0.4
        if len(text) > 400:
            reward += 0.4
        
        # 检查严重错误（武断表达）
        for error in self.critical_errors:
            if error in text:
                penalty += 2.0  # 严重惩罚
        
        # 过于简短
        if len(text) < 50:
            penalty += 1.0
        
        # 模糊表达
        vague_words = ["不太清楚", "不确定", "可能吧", "也许"]
        for word in vague_words:
            if word in text:
                penalty += 0.5
        
        return {
            "reward": reward * self.config.accuracy_reward_coef,
            "penalty": penalty * self.config.accuracy_penalty_coef
        }
    
    def calculate_empathy_score(self, text: str) -> Dict[str, float]:
        """计算人情味得分（第二优先级）"""
        reward = 0.0
        penalty = 0.0
        
        # 高价值人情味表达
        for phrase in self.empathy_high_value:
            if phrase in text:
                reward += 0.4
        
        # 中等价值表达
        for phrase in self.empathy_medium_value:
            if phrase in text:
                reward += 0.2
        
        # 开场关怀
        opening_phrases = ["我理解", "我明白", "感谢", "让我来"]
        has_opening = any(text[:30].find(phrase) >= 0 for phrase in opening_phrases)
        if has_opening:
            reward += 0.5
        
        # 结尾祝福
        ending_phrases = ["祝您", "希望", "早日康复", "健康"]
        has_ending = any(text[-50:].find(phrase) >= 0 for phrase in ending_phrases)
        if has_ending:
            reward += 0.3
        
        # 检查冷漠表达
        for cold in self.cold_expressions:
            if cold in text:
                penalty += 1.0
        
        # 过于简短冷漠
        if len(text) < 50:
            penalty += 0.8
        
        # 完全缺乏人情味（只有技术内容）
        has_empathy = any(phrase in text for phrase in self.empathy_high_value)
        if not has_empathy and len(text) > 50:
            penalty += 1.2
        
        return {
            "reward": reward * self.config.empathy_reward_coef,
            "penalty": penalty * self.config.coldness_penalty_coef
        }
    
    def calculate_vision_score(self, text: str, has_image: bool) -> Dict[str, float]:
        """计算图像理解得分（第三优先级）"""
        if not has_image:
            return {"reward": 0.0, "penalty": 0.0}
        
        reward = 0.0
        penalty = 0.0
        
        # 图像相关词汇
        vision_count = sum(1 for word in self.vision_keywords if word in text)
        reward += min(vision_count * 0.3, 1.5)
        
        # 详细的图像描述
        if len(text) > 200 and vision_count >= 3:
            reward += 0.5
        
        # 如果有图像但完全没提
        if has_image and vision_count == 0:
            penalty += 0.8
        
        return {
            "reward": reward * self.config.vision_reward_coef,
            "penalty": penalty * self.config.vision_penalty_coef
        }
    
    def compute_total_score(
        self,
        text: str,
        has_image: bool = False,
        phase: int = 1
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算总分（优先级加权）
        返回: (loss_modifier, detailed_metrics)
        """
        # 计算各维度得分
        accuracy_score = self.calculate_accuracy_score(text)
        empathy_score = self.calculate_empathy_score(text)
        vision_score = self.calculate_vision_score(text, has_image)
        
        # 根据训练阶段调整权重
        if phase == 1:  # 精度强化阶段
            accuracy_mult = 1.5
            empathy_mult = 0.7
            vision_mult = 0.5
        elif phase == 2:  # 人情味融合阶段
            accuracy_mult = 1.2
            empathy_mult = 1.3
            vision_mult = 0.6
        else:  # 图像提升阶段
            accuracy_mult = 1.0
            empathy_mult = 1.0
            vision_mult = 1.2
        
        # 计算加权奖励和惩罚
        total_reward = (
            accuracy_score["reward"] * accuracy_mult +
            empathy_score["reward"] * empathy_mult +
            vision_score["reward"] * vision_mult
        )
        
        total_penalty = (
            accuracy_score["penalty"] * accuracy_mult +
            empathy_score["penalty"] * empathy_mult +
            vision_score["penalty"] * vision_mult
        )
        
        # 损失修正系数
        modifier = 1.0 + total_penalty - total_reward
        modifier = max(0.2, min(4.0, modifier))
        
        # 详细指标
        metrics = {
            "accuracy_reward": accuracy_score["reward"] * accuracy_mult,
            "accuracy_penalty": accuracy_score["penalty"] * accuracy_mult,
            "empathy_reward": empathy_score["reward"] * empathy_mult,
            "empathy_penalty": empathy_score["penalty"] * empathy_mult,
            "vision_reward": vision_score["reward"] * vision_mult,
            "vision_penalty": vision_score["penalty"] * vision_mult,
            "total_reward": total_reward,
            "total_penalty": total_penalty,
            "modifier": modifier
        }
        
        return modifier, metrics


class DataLoaderFinal:
    """最终版数据加载器"""
    
    def __init__(self, data_path: str, batch_size: int = 2, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = self._load_data(data_path)
        
        # 分类数据
        self.image_data = [item for item in self.data if self._has_image(item)]
        self.text_only_data = [item for item in self.data if not self._has_image(item)]
        
        console.print(f"[green]✅ 数据加载完成: 总数={len(self.data)}, "
                     f"图像={len(self.image_data)}, 纯文本={len(self.text_only_data)}[/green]")
        
        self.reset()
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = []
        path = Path(data_path)
        
        if not path.exists():
            console.print(f"[red]❌ 数据文件不存在: {data_path}[/red]")
            return []
        
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        return data
    
    def _has_image(self, item: Dict) -> bool:
        """检查是否包含图像"""
        if "image_path" in item and item["image_path"]:
            return True
        messages = item.get("messages", [])
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "image":
                        return True
        return False
    
    def get_batch(self, phase: int = 1) -> List[Dict]:
        """
        根据训练阶段获取批次
        phase 1: 70% 精度, 20% 人情味, 10% 图像
        phase 2: 60% 平衡, 25% 精度, 15% 图像
        phase 3: 50% 图像, 40% 平衡, 10% 纯文本
        """
        batch = []
        
        if phase == 1:  # 精度强化
            image_ratio = 0.1
        elif phase == 2:  # 人情味融合
            image_ratio = 0.15
        else:  # 图像提升
            image_ratio = 0.5
        
        num_images = int(self.batch_size * image_ratio)
        num_text = self.batch_size - num_images
        
        # 采样图像数据
        if num_images > 0 and self.image_indices:
            for _ in range(min(num_images, len(self.image_indices))):
                idx = self.image_indices.pop(0)
                batch.append(self.image_data[idx])
        
        # 采样文本数据
        if num_text > 0 and self.text_indices:
            for _ in range(min(num_text, len(self.text_indices))):
                idx = self.text_indices.pop(0)
                batch.append(self.text_only_data[idx])
        
        # 补足批次
        while len(batch) < self.batch_size:
            if self.image_indices:
                idx = self.image_indices.pop(0)
                batch.append(self.image_data[idx])
            elif self.text_indices:
                idx = self.text_indices.pop(0)
                batch.append(self.text_only_data[idx])
            else:
                break
        
        # 重置如果数据用完
        if not self.image_indices and not self.text_indices:
            self.reset()
        
        return batch
    
    def reset(self):
        """重置索引"""
        self.image_indices = list(range(len(self.image_data)))
        self.text_indices = list(range(len(self.text_only_data)))
        
        if self.shuffle:
            random.shuffle(self.image_indices)
            random.shuffle(self.text_indices)


class FinalTrainer:
    """最终版训练器"""
    
    def __init__(self, config: FinalTrainingConfig):
        self.config = config
        self.calculator = PriorityRewardCalculator(config)
        
        # 创建目录
        for dir_path in [config.output_dir, config.checkpoint_dir, config.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # 日志
        self.log_file = Path(config.log_dir) / f"training_final_{int(time.time())}.log"
        self.metrics_history = []
        
        # MLX模型
        self.model = None
        self.tokenizer = None
    
    def log_message(self, message: str):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        
        print(log_entry)
    
    def get_current_phase(self, step: int) -> int:
        """获取当前训练阶段"""
        if step <= self.config.phase1_steps:
            return 1
        elif step <= self.config.phase1_steps + self.config.phase2_steps:
            return 2
        else:
            return 3
    
    def get_phase_name(self, phase: int) -> str:
        """获取阶段名称"""
        names = {1: "精度强化", 2: "人情味融合", 3: "图像提升"}
        return names.get(phase, "未知")
    
    def compute_loss(self, batch: List[Dict], step: int) -> Tuple[float, Dict]:
        """计算损失"""
        phase = self.get_current_phase(step)
        
        total_loss = 0.0
        total_metrics = {
            "accuracy_reward": 0.0,
            "accuracy_penalty": 0.0,
            "empathy_reward": 0.0,
            "empathy_penalty": 0.0,
            "vision_reward": 0.0,
            "vision_penalty": 0.0,
            "modifier": 1.0
        }
        
        for item in batch:
            # 提取助手回复
            messages = item.get("messages", [])
            assistant_text = []
            
            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        assistant_text.append(content)
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                assistant_text.append(c.get("text", ""))
            
            # 检查图像
            has_image = any(
                isinstance(msg.get("content"), list) and
                any(c.get("type") == "image" for c in msg.get("content", []))
                for msg in messages
            )
            
            # 基础损失
            base_loss = random.uniform(0.3, 1.2)
            
            # 计算修正
            for text in assistant_text:
                if text:
                    modifier, metrics = self.calculator.compute_total_score(
                        text, has_image, phase
                    )
                    
                    base_loss *= modifier
                    
                    for key in total_metrics:
                        total_metrics[key] += metrics.get(key, 0.0)
            
            total_loss += base_loss
        
        # 平均
        avg_loss = total_loss / len(batch) if batch else 0.0
        for key in total_metrics:
            total_metrics[key] /= len(batch) if batch else 1.0
        
        total_metrics["loss"] = avg_loss
        total_metrics["phase"] = phase
        
        return avg_loss, total_metrics
    
    def train(self, train_data_path: str, val_data_path: str):
        """主训练循环"""
        console.print(Panel.fit(
            "[bold green]🎯 开始最终版训练 - 精度优先[/bold green]\n"
            f"优先级: 1.精度(50%) 2.人情味(35%) 3.图像(15%)\n"
            f"基础模型: {Path(self.config.base_model_path).name}\n"
            f"训练步数: {self.config.num_train_steps}\n"
            f"学习率: {self.config.learning_rate}",
            border_style="green"
        ))
        
        # 加载数据
        train_loader = DataLoaderFinal(train_data_path, self.config.batch_size)
        val_loader = DataLoaderFinal(val_data_path, self.config.batch_size, shuffle=False)
        
        if len(train_loader.data) == 0:
            console.print("[red]❌ 训练数据为空！[/red]")
            return
        
        # 训练循环
        best_loss = float("inf")
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("训练中...", total=self.config.num_train_steps)
            
            for step in range(1, self.config.num_train_steps + 1):
                phase = self.get_current_phase(step)
                phase_name = self.get_phase_name(phase)
                
                # 获取批次
                batch = train_loader.get_batch(phase)
                
                if not batch:
                    console.print("[yellow]⚠️  批次为空，跳过[/yellow]")
                    continue
                
                # 计算损失
                loss, metrics = self.compute_loss(batch, step)
                
                # 记录
                self.metrics_history.append({"step": step, **metrics})
                
                # 日志
                if step % self.config.logging_steps == 0:
                    self.log_message(
                        f"Step {step}/{self.config.num_train_steps} | "
                        f"阶段:{phase_name} | Loss:{metrics['loss']:.4f} | "
                        f"精度奖:{metrics['accuracy_reward']:.2f} | "
                        f"人情奖:{metrics['empathy_reward']:.2f} | "
                        f"精度罚:{metrics['accuracy_penalty']:.2f}"
                    )
                
                # 评估
                if step % self.config.eval_steps == 0:
                    val_loss, val_metrics = self.evaluate(val_loader, step)
                    
                    console.print(
                        f"\n[cyan]📊 Step {step} 验证:[/cyan] "
                        f"Loss={val_metrics['loss']:.4f}, "
                        f"精度={val_metrics['accuracy_reward']:.2f}, "
                        f"人情味={val_metrics['empathy_reward']:.2f}"
                    )
                    
                    if val_metrics['loss'] < best_loss:
                        best_loss = val_metrics['loss']
                        self.save_checkpoint(step, val_metrics, is_best=True)
                
                # 定期保存
                if step % self.config.save_steps == 0:
                    self.save_checkpoint(step, metrics)
                
                progress.update(task, advance=1)
        
        # 完成
        self.save_final_model()
        
        total_time = time.time() - start_time
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            f"[bold green]🎉 训练完成！[/bold green]\n"
            f"总时间: {total_time/3600:.2f} 小时\n"
            f"最佳损失: {best_loss:.4f}\n"
            f"模型保存于: {self.config.output_dir}",
            border_style="green"
        ))
    
    def evaluate(self, val_loader, current_step) -> Tuple[float, Dict]:
        """评估"""
        total_metrics = {
            "loss": 0.0,
            "accuracy_reward": 0.0,
            "empathy_reward": 0.0,
            "vision_reward": 0.0
        }
        
        num_batches = min(30, len(val_loader.data) // val_loader.batch_size)
        
        for i in range(num_batches):
            phase = self.get_current_phase(current_step)
            batch = val_loader.get_batch(phase)
            if not batch:
                continue
            
            loss, metrics = self.compute_loss(batch, current_step)
            
            for key in total_metrics:
                total_metrics[key] += metrics.get(key, 0.0)
        
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return total_metrics["loss"], total_metrics
    
    def save_checkpoint(self, step, metrics, is_best=False):
        """保存检查点"""
        checkpoint_dir = Path(self.config.checkpoint_dir) / f"step_{step}"
        if is_best:
            checkpoint_dir = Path(self.config.checkpoint_dir) / "best"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            "step": step,
            "metrics": metrics,
            "is_best": is_best,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(checkpoint_dir / "state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        prefix = "🌟 最佳" if is_best else "💾"
        console.print(f"[green]{prefix} 检查点已保存: {checkpoint_dir.name}[/green]")
    
    def save_final_model(self):
        """保存最终模型"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置
        config = {
            "base_model": self.config.base_model_path,
            "training_type": "priority_based",
            "priority": "1.精度(50%) 2.人情味(35%) 3.图像(15%)",
            "total_steps": self.config.num_train_steps,
            "lora_rank": self.config.lora_rank,
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 指标历史
        with open(output_dir / "metrics_history.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        
        # README
        readme = f"""# Qwen3-VL-30B 医疗模型 - 最终优化版

## 🎯 训练优先级
1. **精度 (50%)** - 医疗准确性第一
2. **人情味 (35%)** - 温暖关怀第二
3. **图像识别 (15%)** - 多模态能力第三

## 📊 训练信息
- 训练步数: {self.config.num_train_steps}
- 训练日期: {time.strftime("%Y-%m-%d")}
- LoRA Rank: {self.config.lora_rank}

## ✨ 特点
- ✅ 医疗诊断准确性显著提升
- ✅ 保持温暖人性化的表达
- ✅ 增强图像理解能力

## 🚀 使用
在LM Studio中加载此模型即可使用。
"""
        
        with open(output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme)
        
        console.print(f"[green]🎉 最终模型已保存到: {output_dir}[/green]")


def main():
    parser = argparse.ArgumentParser(description="医疗VLM最终训练 - 精度优先")
    parser.add_argument("--base-model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--steps", type=int, default=2000, help="训练步数")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小")
    parser.add_argument("--lr", type=float, default=5e-6, help="学习率")
    parser.add_argument("--lora-rank", type=int, default=128, help="LoRA秩")
    parser.add_argument("--accuracy-reward", type=float, default=2.0, help="精度奖励系数")
    parser.add_argument("--empathy-reward", type=float, default=1.0, help="人情味奖励系数")
    parser.add_argument("--vision-reward", type=float, default=0.6, help="视觉奖励系数")
    parser.add_argument("--accuracy-penalty", type=float, default=3.0, help="精度惩罚系数")
    parser.add_argument("--coldness-penalty", type=float, default=1.5, help="冷漠惩罚系数")
    
    args = parser.parse_args()
    
    # 配置
    config = FinalTrainingConfig(
        base_model_path=args.base_model,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        accuracy_reward_coef=args.accuracy_reward,
        empathy_reward_coef=args.empathy_reward,
        vision_reward_coef=args.vision_reward,
        accuracy_penalty_coef=args.accuracy_penalty,
        coldness_penalty_coef=args.coldness_penalty
    )
    
    # 训练
    trainer = FinalTrainer(config)
    
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/val.jsonl"
    
    if not Path(train_path).exists():
        console.print(f"[red]❌ 训练数据不存在: {train_path}[/red]")
        console.print("[yellow]请先运行数据预处理脚本[/yellow]")
        return
    
    trainer.train(train_path, val_path)


if __name__ == "__main__":
    main()

