#!/usr/bin/env python3
"""
快速测试药物知识库系统
"""

from drug_knowledge_base import DrugKnowledgeBase
from generate_drug_training_data import DrugTrainingDataGenerator
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def test_case_1():
    """测试案例1：轻度过敏性紫癜"""
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]测试案例 1: 轻度过敏性紫癜[/bold cyan]\n"
        "用户问题：我现在有过敏性紫癜，腿上有一些紫色的小点，不是很严重，我该吃什么药？",
        border_style="cyan"
    ))
    
    generator = DrugTrainingDataGenerator()
    response = generator._generate_response_for_mild_hsp()
    
    console.print("\n[bold green]AI 回复：[/bold green]\n")
    md = Markdown(response)
    console.print(md)


def test_case_2():
    """测试案例2：中度过敏性紫癜伴关节痛"""
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]测试案例 2: 中度过敏性紫癜伴关节痛[/bold cyan]\n"
        "用户问题：我小时候得过过敏性紫癜，现在又复发了，不仅有皮疹，关节也很疼，应该吃什么药？",
        border_style="cyan"
    ))
    
    generator = DrugTrainingDataGenerator()
    response = generator._generate_response_for_moderate_hsp_with_arthritis()
    
    console.print("\n[bold green]AI 回复：[/bold green]\n")
    md = Markdown(response)
    console.print(md)


def test_case_3():
    """测试案例3：重度紫癜性肾炎"""
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]测试案例 3: 重度紫癜性肾炎[/bold cyan]\n"
        "用户问题：我的过敏性紫癜很严重，尿检发现有蛋白尿和血尿，医生说是紫癜性肾炎，需要用什么药？",
        border_style="cyan"
    ))
    
    generator = DrugTrainingDataGenerator()
    response = generator._generate_response_for_severe_hsp_with_nephritis()
    
    console.print("\n[bold green]AI 回复：[/bold green]\n")
    md = Markdown(response)
    console.print(md)


def test_case_4():
    """测试案例4：维持治疗"""
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]测试案例 4: 激素减量维持治疗[/bold cyan]\n"
        "用户问题：我因为紫癜在吃激素（泼尼松），现在病情稳定了，医生说要减量，但又怕复发，有什么维持治疗的药物吗？",
        border_style="cyan"
    ))
    
    generator = DrugTrainingDataGenerator()
    response = generator._generate_response_for_maintenance_therapy()
    
    console.print("\n[bold green]AI 回复：[/bold green]\n")
    md = Markdown(response)
    console.print(md)


def test_drug_search():
    """测试药物搜索功能"""
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]测试：药物搜索功能[/bold cyan]",
        border_style="cyan"
    ))
    
    kb = DrugKnowledgeBase()
    
    # 搜索布洛芬
    drug = kb.search_drug_by_name("布洛芬")
    if drug:
        console.print(f"\n[bold yellow]搜索结果：{drug['name']}[/bold yellow]")
        console.print(kb.format_drug_info(drug))
    
    # 搜索泼尼松
    drug = kb.search_drug_by_name("泼尼松")
    if drug:
        console.print(f"\n[bold yellow]搜索结果：{drug['name']}[/bold yellow]")
        console.print(kb.format_drug_info(drug))


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold green]💊 药物知识库测试系统[/bold green]\n"
        "展示如何从药物类别细化到具体药物建议",
        border_style="green"
    ))
    
    while True:
        console.print("\n[bold blue]请选择测试案例：[/bold blue]")
        console.print("1. 轻度过敏性紫癜（具体药物推荐）")
        console.print("2. 中度过敏性紫癜伴关节痛（分级用药方案）")
        console.print("3. 重度紫癜性肾炎（完整治疗方案）")
        console.print("4. 激素减量维持治疗（长期管理）")
        console.print("5. 药物搜索功能测试")
        console.print("6. 查看所有药物类别")
        console.print("0. 退出")
        
        choice = input("\n请输入选项（0-6）: ").strip()
        
        if choice == "1":
            test_case_1()
        elif choice == "2":
            test_case_2()
        elif choice == "3":
            test_case_3()
        elif choice == "4":
            test_case_4()
        elif choice == "5":
            test_drug_search()
        elif choice == "6":
            kb = DrugKnowledgeBase()
            console.print("\n[bold blue]📋 所有药物类别：[/bold blue]\n")
            for i, category in enumerate(kb.get_all_categories(), 1):
                drugs = kb.get_drugs_by_category(category)
                console.print(f"{i}. [bold]{category}[/bold] ({len(drugs)} 种药物)")
                for drug in drugs:
                    console.print(f"   - {drug['name']}（{drug['generic_name']}）")
        elif choice == "0":
            console.print("\n[green]感谢使用！再见！[/green]")
            break
        else:
            console.print("[red]无效选项，请重新选择[/red]")
        
        if choice in ["1", "2", "3", "4", "5"]:
            input("\n按回车键继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]程序已中断[/yellow]")
    except Exception as e:
        console.print(f"\n[red]错误: {str(e)}[/red]")



