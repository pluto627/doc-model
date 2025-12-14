#!/usr/bin/env python3
"""
准备 V4.1 训练数据
整合：现有中文医学数据 + 英文医学影像数据 + 身份认知数据
目标：提升图像识别精准度 + 保持问答精准度 + 人情味
"""

import json
import random
from pathlib import Path
from datasets import load_dataset
from rich.console import Console

console = Console()

# 输出目录
OUTPUT_DIR = Path("data_v41")
OUTPUT_DIR.mkdir(exist_ok=True)

# 人情味回复模板
EMPATHY_PREFIXES_CN = [
    "感谢您的咨询。",
    "我来帮您分析一下。",
    "让我仔细看看。",
    "根据您的描述，",
    "我理解您的担忧，",
]

EMPATHY_SUFFIXES_CN = [
    "\n\n如有疑问，请随时告诉我，我会尽力帮您解答~",
    "\n\n建议您与主治医生进一步讨论。健康无小事，我们一起关注！",
    "\n\n希望这些信息对您有帮助。有什么不明白的地方随时问我哦~",
    "\n\n祝您早日康复！身体健康最重要~",
    "\n\n有任何问题都可以继续咨询，我随时都在！",
    "\n\n保持好心情也是养生的一部分哦，有问题随时找我~",
    "\n\n记得好好休息，照顾好自己！有问题再来找我~",
]

EMPATHY_PREFIXES_EN = [
    "Based on the medical image analysis, ",
    "After careful examination, ",
    "Looking at the imaging findings, ",
    "The radiological assessment shows ",
    "Upon reviewing the scan, ",
]

EMPATHY_SUFFIXES_EN = [
    "\n\nPlease consult with your physician for further evaluation.",
    "\n\nI hope this analysis is helpful.",
    "\n\nFeel free to ask if you have any questions.",
]


def add_empathy_cn(answer: str) -> str:
    """添加中文人情味"""
    if random.random() < 0.7:  # 70%概率添加
        prefix = random.choice(EMPATHY_PREFIXES_CN)
        suffix = random.choice(EMPATHY_SUFFIXES_CN)
        return f"{prefix}{answer}{suffix}"
    return answer


def add_empathy_en(answer: str) -> str:
    """添加英文专业表达"""
    if random.random() < 0.5:
        prefix = random.choice(EMPATHY_PREFIXES_EN)
        suffix = random.choice(EMPATHY_SUFFIXES_EN)
        return f"{prefix}{answer}{suffix}"
    return answer


def load_existing_processed_data():
    """加载现有的已处理数据（中文医学问答）"""
    console.print("\n[cyan]📥 加载现有中文医学数据...[/cyan]")
    
    samples = []
    
    # 加载 train.jsonl
    train_path = Path("data/processed/train.jsonl")
    if train_path.exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    samples.append(data)
                except:
                    continue
    
    # 加载 drug_training_data.jsonl
    drug_path = Path("data/processed/drug_training_data.jsonl")
    if drug_path.exists():
        with open(drug_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    samples.append(data)
                except:
                    continue
    
    console.print(f"  ✅ 加载了 {len(samples)} 条现有中文数据")
    return samples


def load_existing_mlx_data():
    """加载 data_mlx 中的数据"""
    console.print("\n[cyan]📥 加载 data_mlx 数据...[/cyan]")
    
    samples = []
    mlx_train = Path("data_mlx/train.jsonl")
    
    if mlx_train.exists():
        with open(mlx_train, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    samples.append(json.loads(line))
                except:
                    continue
    
    console.print(f"  ✅ 加载了 {len(samples)} 条 MLX 数据")
    return samples


def process_vqa_rad():
    """处理 VQA-RAD 数据集（医学影像问答）"""
    console.print("\n[cyan]📥 处理 VQA-RAD 医学影像数据...[/cyan]")
    
    try:
        dataset = load_dataset("flaviagiammarino/vqa-rad")
        samples = []
        
        for item in dataset['train']:
            question = item['question']
            answer = item['answer']
            
            # 增强回答
            enhanced = enhance_radiology_answer(question, answer)
            
            sample = {
                "messages": [
                    {"role": "user", "content": f"Please analyze this medical image: {question}"},
                    {"role": "assistant", "content": enhanced}
                ]
            }
            samples.append(sample)
        
        for item in dataset['test']:
            question = item['question']
            answer = item['answer']
            enhanced = enhance_radiology_answer(question, answer)
            
            sample = {
                "messages": [
                    {"role": "user", "content": f"Analyze the imaging finding: {question}"},
                    {"role": "assistant", "content": enhanced}
                ]
            }
            samples.append(sample)
        
        console.print(f"  ✅ 处理了 {len(samples)} 条 VQA-RAD 样本")
        return samples
    except Exception as e:
        console.print(f"  ❌ VQA-RAD 错误: {e}")
        return []


def process_medical_vqa():
    """处理 Medical-VQA 数据集"""
    console.print("\n[cyan]📥 处理 Medical-VQA 数据...[/cyan]")
    
    try:
        dataset = load_dataset("rbojja/medical-vqa")
        samples = []
        
        for item in dataset['train']:
            try:
                conversations = item.get('conversations', [])
                if len(conversations) >= 2:
                    user_msg = ""
                    assistant_msg = ""
                    
                    for conv in conversations:
                        if conv.get('from') == 'human':
                            user_msg = conv.get('value', '')
                        elif conv.get('from') == 'gpt':
                            assistant_msg = conv.get('value', '')
                    
                    if user_msg and assistant_msg:
                        enhanced = add_empathy_en(assistant_msg)
                        sample = {
                            "messages": [
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": enhanced}
                            ]
                        }
                        samples.append(sample)
            except:
                continue
        
        console.print(f"  ✅ 处理了 {len(samples)} 条 Medical-VQA 样本")
        return samples
    except Exception as e:
        console.print(f"  ❌ Medical-VQA 错误: {e}")
        return []


def process_medical_multimodal():
    """处理 Medical Multimodal 数据集"""
    console.print("\n[cyan]📥 处理 Medical Multimodal 数据...[/cyan]")
    
    try:
        dataset = load_dataset("FreedomIntelligence/Medical_Multimodal_Evaluation_Data")
        samples = []
        
        for item in dataset['test']:
            try:
                question = item.get('question', '')
                answer = item.get('answer', '')
                options = item.get('options', [])
                
                if question and answer:
                    # 构建详细回答
                    if options and len(options) > 0:
                        detailed = f"Based on the medical image analysis, the answer is {answer}."
                    else:
                        detailed = f"The imaging analysis indicates: {answer}"
                    
                    enhanced = add_empathy_en(detailed)
                    
                    sample = {
                        "messages": [
                            {"role": "user", "content": f"Please analyze this medical image and answer: {question}"},
                            {"role": "assistant", "content": enhanced}
                        ]
                    }
                    samples.append(sample)
            except:
                continue
        
        console.print(f"  ✅ 处理了 {len(samples)} 条 Medical Multimodal 样本")
        return samples
    except Exception as e:
        console.print(f"  ❌ Medical Multimodal 错误: {e}")
        return []


def enhance_radiology_answer(question: str, answer: str) -> str:
    """增强放射学回答"""
    q_lower = question.lower()
    
    # 根据问题类型添加专业前缀
    if any(kw in q_lower for kw in ['ct', 'computed tomography']):
        prefix = "On CT imaging, "
    elif any(kw in q_lower for kw in ['mri', 'magnetic resonance']):
        prefix = "On MRI evaluation, "
    elif any(kw in q_lower for kw in ['x-ray', 'xray', 'radiograph']):
        prefix = "On radiographic examination, "
    elif any(kw in q_lower for kw in ['ultrasound', 'sonography']):
        prefix = "On ultrasonographic assessment, "
    elif any(kw in q_lower for kw in ['brain', 'cerebral', 'head']):
        prefix = "Neuroimaging reveals "
    elif any(kw in q_lower for kw in ['chest', 'lung', 'pulmonary']):
        prefix = "Chest imaging demonstrates "
    elif any(kw in q_lower for kw in ['abdomen', 'liver', 'kidney']):
        prefix = "Abdominal imaging shows "
    else:
        prefix = "Medical imaging analysis indicates "
    
    # 扩展简短答案
    if len(answer) < 30:
        if answer.lower() in ['yes', 'no']:
            if answer.lower() == 'yes':
                answer = "yes, the findings are present as indicated."
            else:
                answer = "no, the findings are not evident on imaging."
        enhanced = f"{prefix}{answer}"
    else:
        enhanced = f"{prefix}{answer}"
    
    return add_empathy_en(enhanced)


def create_identity_samples():
    """创建身份认知样本（增强版）"""
    console.print("\n[cyan]📝 创建身份认知样本...[/cyan]")
    
    identity_qa = [
        # 中文身份（带趣味性）
        ("你是谁？", "嗨！我是XunDoc，您的贴心健康小助手~😊 我可以帮您解答健康问题、分析医学影像、提供专业的健康指导。有什么可以帮您的吗？"),
        ("你叫什么名字？", "我叫XunDoc，您可以叫我小寻~是一个热爱医学、喜欢帮助人的智能健康助手！有什么健康问题尽管问我~"),
        ("介绍一下你自己", "很高兴认识您！我是XunDoc智能健康助手，大家也叫我小寻~\n\n我的特长是：\n1. 🔬 解答各类健康和医学问题\n2. 🩻 分析医学影像（X光、CT、MRI等）\n3. 💊 提供健康指导和生活建议\n\n虽然我是AI，但我会用心回答您的每一个问题！有什么想咨询的吗？"),
        ("你是什么模型？", "我是XunDoc，一个专注于健康医疗领域的智能助手~您可以把我当成一个随时在线的健康顾问，有问题随时找我聊聊~"),
        ("你能做什么？", "作为您的健康小管家，我可以帮您：\n\n1. 🩻 **医学影像分析**：帮您解读X光、CT、MRI等检查结果\n2. 💬 **健康问答**：回答各种健康和医学问题\n3. 🌟 **健康指导**：提供贴心的健康建议\n\n有什么健康问题想聊聊吗？我随时都在~"),
        ("你好", "您好呀！我是XunDoc，您的智能健康小助手~今天有什么可以帮您的吗？😊"),
        ("请问你是哪个公司开发的", "我是XunDoc智能健康助手，专注于医疗健康领域~我的使命是用专业知识帮助每一位用户！有什么健康问题想咨询吗？"),
        ("早上好", "早上好呀！新的一天，希望您元气满满~我是XunDoc，有什么健康问题可以帮您解答吗？"),
        ("晚上好", "晚上好！忙碌了一天，记得好好休息哦~我是XunDoc，有什么健康问题想咨询吗？"),
        ("谢谢", "不客气~能帮到您是我的荣幸！如果还有其他问题，随时找我哦~祝您健康快乐！😊"),
        ("再见", "再见啦！记得保持好心情，有问题随时回来找我~祝您身体健康！👋"),
        
        # 英文身份
        ("Who are you?", "Hi there! I'm XunDoc, your friendly health assistant! 😊 I can help you with medical questions, analyze medical images, and provide professional health guidance. How can I assist you today?"),
        ("What is your name?", "My name is XunDoc! I'm a professional health assistant with expertise in medical knowledge and imaging analysis. Feel free to ask me anything~"),
        ("Introduce yourself", "Nice to meet you! I'm XunDoc, an AI health assistant specialized in:\n\n1. 🔬 Medical image analysis (X-ray, CT, MRI)\n2. 💬 Health Q&A and medical consultations\n3. 🌟 Professional health guidance\n\nI'm here to help with any health concerns you may have!"),
        ("Hello", "Hello! I'm XunDoc, your intelligent health assistant. What health questions can I help you with today? 😊"),
        ("Thank you", "You're welcome! I'm glad I could help. If you have any more questions, feel free to ask anytime! Stay healthy! 😊"),
    ]
    
    samples = []
    for q, a in identity_qa:
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    
    # 复制多次加强记忆（每条复制30次）
    samples = samples * 30
    random.shuffle(samples)
    
    console.print(f"  ✅ 创建了 {len(samples)} 条身份认知样本")
    return samples


def create_precision_qa_samples():
    """创建高精准度医学问答样本"""
    console.print("\n[cyan]📝 创建精准度问答样本...[/cyan]")
    
    precision_qa = [
        # 用药时间
        ("感冒药应该什么时候吃？", "感冒药的服用时间因药物类型而异：\n\n1. **解热镇痛药（如布洛芬）**：建议饭后服用，减少胃肠刺激\n2. **抗组胺药（如氯雷他定）**：每日一次，建议早晨或睡前服用\n3. **复方感冒药**：按说明书规定时间，通常每4-6小时一次\n\n建议您查看具体药品说明书，或咨询药师获取准确用药指导。"),
        
        ("降压药什么时候吃效果最好？", "降压药的最佳服用时间：\n\n1. **长效降压药（如氨氯地平）**：建议早晨服用，每日一次\n2. **短效降压药**：根据血压监测结果调整，通常早晚各一次\n3. **利尿类降压药**：建议早晨服用，避免夜间频繁起夜\n\n建议您监测血压变化，并遵医嘱调整用药时间。"),
        
        ("血糖什么时候测量最准确？", "血糖监测的最佳时间点：\n\n1. **空腹血糖**：早晨起床后、进食前测量（禁食8-10小时后）\n2. **餐后血糖**：从第一口进食开始计时2小时后测量\n3. **随机血糖**：任意时间点测量\n\n正常参考值：\n- 空腹：3.9-6.1 mmol/L\n- 餐后2小时：< 7.8 mmol/L\n\n建议定期记录血糖变化，便于医生调整治疗方案。"),
        
        # 处理方法
        ("发烧了应该怎么处理？", "发烧的处理建议：\n\n**物理降温**：\n- 温水擦浴（水温32-34℃）\n- 适当减少衣物\n- 保持室内通风\n- 多喝温水\n\n**药物降温**（体温≥38.5℃时）：\n- 成人：对乙酰氨基酚500mg或布洛芬400mg\n- 儿童：按体重计算剂量\n\n**及时就医情况**：\n- 体温持续超过39℃\n- 发热超过3天不退\n- 伴有剧烈头痛、皮疹、抽搐等症状\n\n请注意休息，密切观察体温变化。"),
        
        ("血压高了怎么办？", "血压升高的应对措施：\n\n**即时处理**：\n1. 保持冷静，找安静处休息\n2. 深呼吸，放松身心\n3. 服用常规降压药（如有）\n\n**生活调整**：\n- 低盐饮食（每日钠<6g）\n- 规律运动（每周150分钟中等强度）\n- 限制饮酒\n- 控制体重\n- 保证充足睡眠\n\n**就医指征**：\n- 收缩压≥180mmHg或舒张压≥110mmHg\n- 伴有头痛、胸闷、视物模糊\n- 药物控制不佳\n\n建议您定期监测血压并遵医嘱治疗。"),
    ]
    
    samples = []
    for q, a in precision_qa:
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    
    # 复制增强
    samples = samples * 15
    
    console.print(f"  ✅ 创建了 {len(samples)} 条精准度问答样本")
    return samples


def main():
    console.print("\n" + "="*60)
    console.print("[bold cyan]🏥 准备 V4.1 训练数据[/bold cyan]")
    console.print("[bold]目标：图像识别↑ + 问答精准度↑ + 人情味保持[/bold]")
    console.print("="*60)
    
    all_samples = []
    
    # 1. 加载现有中文数据
    existing_cn = load_existing_processed_data()
    all_samples.extend(existing_cn)
    
    # 2. 加载 MLX 数据
    mlx_data = load_existing_mlx_data()
    all_samples.extend(mlx_data)
    
    # 3. 加载英文医学影像数据
    vqa_rad = process_vqa_rad()
    all_samples.extend(vqa_rad)
    
    medical_vqa = process_medical_vqa()
    all_samples.extend(medical_vqa)
    
    medical_mm = process_medical_multimodal()
    all_samples.extend(medical_mm)
    
    # 4. 添加身份认知样本
    identity = create_identity_samples()
    all_samples.extend(identity)
    
    # 5. 添加精准度问答样本
    precision = create_precision_qa_samples()
    all_samples.extend(precision)
    
    # 6. 去重
    seen = set()
    unique_samples = []
    for s in all_samples:
        key = json.dumps(s, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)
    
    console.print(f"\n[yellow]去重后样本数: {len(unique_samples)}[/yellow]")
    
    # 7. 打乱数据
    random.shuffle(unique_samples)
    
    # 8. 分割数据集
    total = len(unique_samples)
    train_size = int(total * 0.9)
    valid_size = int(total * 0.05)
    
    train_data = unique_samples[:train_size]
    valid_data = unique_samples[train_size:train_size+valid_size]
    test_data = unique_samples[train_size+valid_size:]
    
    # 9. 保存数据
    console.print("\n[cyan]💾 保存数据...[/cyan]")
    
    with open(OUTPUT_DIR / "train.jsonl", 'w', encoding='utf-8') as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    with open(OUTPUT_DIR / "valid.jsonl", 'w', encoding='utf-8') as f:
        for sample in valid_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    with open(OUTPUT_DIR / "test.jsonl", 'w', encoding='utf-8') as f:
        for sample in test_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # 10. 统计
    console.print("\n" + "="*60)
    console.print("[bold green]✅ V4.1 数据准备完成！[/bold green]")
    console.print("="*60)
    console.print(f"\n📊 数据统计：")
    console.print(f"  训练集: {len(train_data):,} 条")
    console.print(f"  验证集: {len(valid_data):,} 条")
    console.print(f"  测试集: {len(test_data):,} 条")
    console.print(f"  总计: {total:,} 条")
    console.print(f"\n📁 保存位置: {OUTPUT_DIR}/")
    
    # 建议训练步数
    steps = min(2000, max(1000, len(train_data) // 10))
    console.print(f"\n💡 建议训练步数: {steps} 步")


if __name__ == "__main__":
    main()
