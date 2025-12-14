#!/usr/bin/env python3
"""
V3模型评估脚本
测试V3模型的精准度和人情味表现
"""
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# MLX导入
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    console.print("[red]❌ MLX未安装[/red]")
    sys.exit(1)


# 测试用例
TEST_CASES = [
    {
        "name": "血压咨询（需要精准数值）",
        "prompt": "我的血压是145/95 mmHg，这个数值正常吗？需要吃药吗？",
        "expected": "精准数值范围、具体建议、保持温暖"
    },
    {
        "name": "血糖检查（需要详细分析）",
        "prompt": "空腹血糖7.2 mmol/L，餐后2小时11.5 mmol/L，这是糖尿病吗？",
        "expected": "准确诊断标准、数值分析、具体治疗建议"
    },
    {
        "name": "药物咨询（需要具体方案）",
        "prompt": "我有高血压，医生开了降压药，什么时候吃效果最好？",
        "expected": "具体时间建议、用药注意事项、保持关怀"
    },
    {
        "name": "症状分析（需要综合判断）",
        "prompt": "最近头晕、乏力、食欲不振，可能是什么问题？",
        "expected": "可能诊断、建议检查项目、温暖安慰"
    },
    {
        "name": "情绪支持（需要人情味）",
        "prompt": "确诊糖尿病后我很焦虑，担心以后的生活...",
        "expected": "同理心、情绪支持、实际建议"
    }
]


def load_v3_model():
    """加载V3模型"""
    # 尝试加载融合版
    fused_path = "/Users/plutoguo/.lmstudio/models/local/Qwen3-VL-30B-Medical-V3-Precision"
    adapter_path = "adapters_v3_precision"
    base_model_path = "/Users/plutoguo/.lmstudio/models/local/Qwen3-VL-30B-Medical-V2-Fused"
    
    console.print("[cyan]🔧 加载V3模型...[/cyan]")
    
    # 先尝试融合版
    if Path(fused_path).exists():
        console.print(f"[green]✅ 使用融合版: {fused_path}[/green]")
        try:
            model, tokenizer = load(fused_path)
            return model, tokenizer, "融合版"
        except Exception as e:
            console.print(f"[yellow]⚠️  融合版加载失败: {e}[/yellow]")
    
    # 尝试adapter版
    if Path(adapter_path).exists() and Path(base_model_path).exists():
        console.print(f"[green]✅ 使用Adapter版[/green]")
        try:
            model, tokenizer = load(base_model_path, adapter_path=adapter_path)
            return model, tokenizer, "Adapter版"
        except Exception as e:
            console.print(f"[yellow]⚠️  Adapter版加载失败: {e}[/yellow]")
    
    console.print("[red]❌ 无法加载V3模型[/red]")
    return None, None, None


def evaluate_response(prompt: str, response: str) -> dict:
    """评估回复质量"""
    scores = {
        "medical_terms": 0,
        "numerical_precision": 0,
        "specificity": 0,
        "empathy": 0,
        "warmth": 0
    }
    
    # 医学术语
    medical_terms = [
        "高血压", "糖尿病", "血压", "血糖", "诊断", "治疗",
        "药物", "检查", "指标", "数值", "范围", "正常值"
    ]
    scores["medical_terms"] = sum(1 for term in medical_terms if term in response)
    
    # 数值精度
    import re
    numbers = re.findall(r'\d+\.?\d*', response)
    units = ["mmHg", "mmol/L", "mg/dL", "U/L", "g/L"]
    has_units = any(unit in response for unit in units)
    scores["numerical_precision"] = len(numbers) + (2 if has_units else 0)
    
    # 具体性
    specific_words = ["具体", "建议", "方案", "步骤", "首先", "其次", "然后"]
    scores["specificity"] = sum(1 for word in specific_words if word in response)
    
    # 人情味
    empathy_words = ["理解", "担心", "焦虑", "关心", "支持"]
    scores["empathy"] = sum(1 for word in empathy_words if word in response)
    
    warmth_words = ["您", "请", "希望", "祝", "陪伴"]
    scores["warmth"] = sum(1 for word in warmth_words if word in response)
    
    return scores


def run_evaluation():
    """运行评估"""
    console.print(Panel.fit(
        "[bold cyan]🧪 V3模型评估工具[/bold cyan]\n"
        "测试精准度和人情味表现",
        border_style="cyan"
    ))
    console.print()
    
    # 加载模型
    model, tokenizer, version = load_v3_model()
    if model is None:
        console.print("[red]❌ 模型加载失败，无法评估[/red]")
        return
    
    console.print(f"[green]✅ 模型已加载 ({version})[/green]")
    console.print()
    
    # 运行测试
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        console.print(f"[cyan]{'='*60}[/cyan]")
        console.print(f"[bold]测试 {i}/{len(TEST_CASES)}: {test_case['name']}[/bold]")
        console.print(f"[yellow]问题:[/yellow] {test_case['prompt']}")
        console.print()
        
        # 生成回复
        try:
            response = generate(
                model,
                tokenizer,
                prompt=test_case['prompt'],
                max_tokens=400,
                verbose=False
            )
            
            console.print(f"[green]回复:[/green]")
            console.print(response)
            console.print()
            
            # 评估
            scores = evaluate_response(test_case['prompt'], response)
            console.print(f"[blue]评分:[/blue]")
            console.print(f"  - 医学术语: {scores['medical_terms']}")
            console.print(f"  - 数值精度: {scores['numerical_precision']}")
            console.print(f"  - 具体性: {scores['specificity']}")
            console.print(f"  - 同理心: {scores['empathy']}")
            console.print(f"  - 温暖度: {scores['warmth']}")
            
            results.append({
                "name": test_case['name'],
                "prompt": test_case['prompt'],
                "response": response,
                "scores": scores
            })
            
        except Exception as e:
            console.print(f"[red]❌ 生成失败: {e}[/red]")
        
        console.print()
    
    # 生成总结
    console.print("[cyan]" + "="*60 + "[/cyan]")
    console.print("[bold green]📊 评估总结[/bold green]")
    console.print()
    
    if results:
        # 创建表格
        table = Table(title="V3模型评估结果")
        table.add_column("测试用例", style="cyan")
        table.add_column("医学术语", justify="center")
        table.add_column("数值精度", justify="center")
        table.add_column("具体性", justify="center")
        table.add_column("同理心", justify="center")
        table.add_column("温暖度", justify="center")
        
        avg_scores = {
            "medical_terms": 0,
            "numerical_precision": 0,
            "specificity": 0,
            "empathy": 0,
            "warmth": 0
        }
        
        for result in results:
            scores = result['scores']
            table.add_row(
                result['name'],
                str(scores['medical_terms']),
                str(scores['numerical_precision']),
                str(scores['specificity']),
                str(scores['empathy']),
                str(scores['warmth'])
            )
            
            for key in avg_scores:
                avg_scores[key] += scores[key]
        
        # 计算平均值
        n = len(results)
        for key in avg_scores:
            avg_scores[key] /= n
        
        table.add_row(
            "[bold]平均分[/bold]",
            f"[bold]{avg_scores['medical_terms']:.1f}[/bold]",
            f"[bold]{avg_scores['numerical_precision']:.1f}[/bold]",
            f"[bold]{avg_scores['specificity']:.1f}[/bold]",
            f"[bold]{avg_scores['empathy']:.1f}[/bold]",
            f"[bold]{avg_scores['warmth']:.1f}[/bold]"
        )
        
        console.print(table)
        console.print()
        
        # 综合评价
        precision_score = (
            avg_scores['medical_terms'] + 
            avg_scores['numerical_precision'] + 
            avg_scores['specificity']
        ) / 3
        
        empathy_score = (
            avg_scores['empathy'] + 
            avg_scores['warmth']
        ) / 2
        
        console.print(f"[bold cyan]综合评价:[/bold cyan]")
        console.print(f"  精准度得分: [green]{precision_score:.2f}[/green]")
        console.print(f"  人情味得分: [green]{empathy_score:.2f}[/green]")
        console.print(f"  综合得分: [bold green]{(precision_score + empathy_score) / 2:.2f}[/bold green]")


def main():
    """主函数"""
    run_evaluation()


if __name__ == "__main__":
    main()



