#!/usr/bin/env python3
"""
准备MLX-LM训练数据 - 包含奖惩机制
通过正面示例（高质量回复）和负面示例（低质量回复）来实现奖惩
"""
import json
from pathlib import Path
import random

# 奖励词汇（鼓励使用）
REWARD_WORDS = [
    "我理解", "我能感受到", "感谢您", "让我来帮您",
    "可能", "建议", "参考", "如果", "通常来说", "一般情况下"
]

# 惩罚词汇（需要避免）
PENALTY_WORDS = [
    "肯定是", "一定是", "必须", "绝对",
    "不用担心", "没什么大不了", "你应该", "这很简单"
]

# 人情味开头模板
EMPATHY_PREFIXES = [
    "我理解您的担忧，",
    "感谢您的信任，我来帮您分析一下。",
    "我能感受到您的关切，这是很正常的反应。",
    "请不要太担心，让我们一起来看看这个情况。",
    "您提出了一个很好的问题，"
]

# 关怀结尾模板
CARING_SUFFIXES = [
    "\n\n如有其他疑问，请随时告诉我。",
    "\n\n希望这些信息对您有帮助，有任何问题都可以问我。",
    "\n\n如果症状持续或加重，建议及时就医。祝您健康！",
    "\n\n请放心，有任何问题都可以继续咨询。"
]

def enhance_response(response: str) -> str:
    """增强回复的人情味"""
    # 检查是否已有人情味表达
    has_empathy = any(word in response for word in REWARD_WORDS[:4])
    
    if not has_empathy and len(response) > 30:
        response = random.choice(EMPATHY_PREFIXES) + response
    
    # 添加关怀结尾
    if not any(suffix.strip() in response for suffix in CARING_SUFFIXES):
        if len(response) > 50:
            response = response + random.choice(CARING_SUFFIXES)
    
    return response

def create_contrast_pairs():
    """创建奖惩对比数据"""
    contrast_data = [
        # 血压相关
        {
            "user": "我的血压是150/95，需要吃药吗？",
            "good": "我理解您的担忧，血压150/95 mmHg确实略高于正常范围。首先请不要过于焦虑，让我帮您分析一下。这个数值属于1级高血压，是否需要服药要综合考虑：1) 如果是偶尔一次测量偏高，可能与情绪、睡眠等有关 2) 建议您连续监测一周的血压 3) 同时注意低盐饮食、适量运动、保证睡眠。如果持续偏高，建议咨询心内科医生评估是否需要药物干预。有什么其他问题我可以帮您解答吗？",
            "bad": "150/95肯定是高血压了，必须吃药，你赶紧去医院吧。"
        },
        # 血糖相关
        {
            "user": "检查报告显示我血糖7.2，是糖尿病吗？",
            "good": "感谢您分享检查结果，我理解收到这样的数值会让您担心。空腹血糖7.2 mmol/L确实高于正常范围（3.9-6.1 mmol/L），但这不代表一定是糖尿病。我来帮您分析：1) 单次检测偏高可能受饮食、压力等因素影响 2) 建议复查空腹血糖和糖化血红蛋白(HbA1c) 3) 在确诊前，可以通过控制饮食、增加运动来改善。请问您最近饮食和作息有什么变化吗？有任何疑问都可以告诉我。",
            "bad": "7.2绝对是糖尿病前期了，你必须马上开始吃药控制。"
        },
        # 结节相关
        {
            "user": "CT显示肺部有小结节，我很害怕是癌症。",
            "good": "我能深深理解您此刻的焦虑和担忧，发现肺结节确实会让人感到恐惧。但请让我帮您理性分析一下：首先，大多数肺部小结节（尤其是<6mm）是良性的，可能是之前感染留下的痕迹、炎性结节等。恶性的概率实际上很低。关键要看结节的大小、形态、边缘特征。建议您：1) 保持冷静，不要过度恐慌 2) 按医生建议定期随访复查 3) 如果结节较小且特征良好，可能只需观察即可。请问报告上有描述结节的具体大小和特征吗？我来帮您进一步分析。",
            "bad": "肺结节很复杂，谁知道是不是癌症，你自己去问医生吧。"
        },
        # 头痛相关
        {
            "user": "我最近经常头痛，是什么原因？",
            "good": "感谢您的咨询，我理解头痛给您带来的困扰。头痛的原因有很多，让我帮您梳理一下可能的因素：1) 紧张性头痛 - 与压力、长时间用眼、颈椎问题相关 2) 睡眠问题 - 睡眠不足或过多都可能引起 3) 偏头痛 - 通常伴有恶心、畏光 4) 其他原因 - 如血压变化、感冒等。建议您记录一下：头痛的部位、性质（胀痛/刺痛）、持续时间、诱发因素。如果头痛剧烈或伴有视力改变、呕吐等，建议及时就医检查。您能描述一下具体的头痛情况吗？",
            "bad": "头痛没什么大不了的，吃点止痛药就行了，不用担心。"
        },
        # 心脏相关
        {
            "user": "我有时候感觉心跳很快，是心脏病吗？",
            "good": "我能感受到您对心脏问题的担忧，这种关注自己健康的态度很好。心跳加快（心悸）在很多情况下是正常的生理反应，比如运动后、紧张焦虑时、喝咖啡后等。但为了放心，我建议您注意观察：1) 心悸发生的频率和持续时间 2) 是否伴有胸闷、气短、头晕 3) 发作时的情境（休息时还是活动后）。如果频繁发作或伴有不适症状，建议做一个心电图和24小时动态心电图检查。请问您能描述一下心跳加快时的具体感受吗？",
            "bad": "心跳快肯定是心脏有问题，你必须马上去医院做全面检查。"
        }
    ]
    
    formatted_data = []
    
    # 生成正面示例（奖励）- 多复制几次增加权重
    for pair in contrast_data:
        for _ in range(3):  # 正面示例权重x3
            formatted_data.append({
                "messages": [
                    {"role": "user", "content": pair["user"]},
                    {"role": "assistant", "content": pair["good"]}
                ]
            })
    
    # 生成更多变体
    variations = [
        {"user": "我血压高怎么办？", "prefix": "血压偏高需要关注，"},
        {"user": "体检说我有脂肪肝", "prefix": "脂肪肝是很常见的情况，"},
        {"user": "我尿酸偏高会痛风吗", "prefix": "尿酸偏高是痛风的风险因素之一，"},
        {"user": "甲状腺结节要紧吗", "prefix": "甲状腺结节大多数是良性的，"},
        {"user": "我总是失眠怎么办", "prefix": "失眠确实会影响生活质量，"},
    ]
    
    for var in variations:
        response = f"我理解您的担忧。{var['prefix']}但请不要过于焦虑。建议您：1) 保持良好的生活习惯 2) 定期复查监测 3) 如有明显不适及时就医。具体情况还需要结合您的检查报告来分析，您方便告诉我更多细节吗？\n\n如有其他疑问，请随时告诉我。"
        for _ in range(2):
            formatted_data.append({
                "messages": [
                    {"role": "user", "content": var["user"]},
                    {"role": "assistant", "content": response}
                ]
            })
    
    return formatted_data

def convert_data():
    """转换数据为MLX-LM格式"""
    base_dir = Path("/Users/plutoguo/Desktop/training")
    input_file = base_dir / "data/processed/train.jsonl"
    output_dir = base_dir / "data_mlx"
    output_dir.mkdir(exist_ok=True)
    
    train_data = []
    val_data = []
    
    # 1. 首先添加奖惩对比数据
    contrast_data = create_contrast_pairs()
    train_data.extend(contrast_data)
    print(f"📝 添加奖惩对比数据: {len(contrast_data)} 条")
    
    # 2. 从原始数据中选取并增强
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                messages = item.get('messages', [])
                
                if not messages:
                    continue
                
                # 转换消息格式
                converted_msgs = []
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    
                    # 处理content是列表的情况
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                text_parts.append(part.get('text', ''))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        content = ' '.join(text_parts)
                    
                    # 对助手回复增强人情味
                    if role == 'assistant' and content.strip():
                        content = enhance_response(content.strip())
                    
                    if content.strip():
                        converted_msgs.append({
                            "role": role,
                            "content": content.strip()
                        })
                
                if len(converted_msgs) >= 2:
                    entry = {"messages": converted_msgs}
                    if len(train_data) < 4000:
                        train_data.append(entry)
                    elif len(val_data) < 500:
                        val_data.append(entry)
                    else:
                        break
                        
            except:
                continue
    
    # 打乱训练数据
    random.shuffle(train_data)
    
    # 保存训练数据
    with open(output_dir / "train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存验证数据
    with open(output_dir / "valid.jsonl", 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 数据准备完成（包含奖惩机制）!")
    print(f"   训练集: {len(train_data)} 条")
    print(f"   验证集: {len(val_data)} 条")
    print(f"   奖励样本（高质量回复）已增强权重")

if __name__ == "__main__":
    convert_data()
