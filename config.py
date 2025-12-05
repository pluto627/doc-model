"""
医疗视觉语言模型训练配置文件
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "finetuned_model")

# 原始模型路径
SOURCE_MODEL_PATH = "/Users/plutoguo/.lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit"

# 数据集配置
DATASETS = {
    "medical_vision_llm": {
        "repo_id": "robailleo/medical-vision-llm-dataset",
        "split": "train",
        "max_samples": 5000
    },
    "aquiles_medical_vision": {
        "repo_id": "Aquiles-ai/Medical-Vision",
        "split": "train",
        "max_samples": 5000
    },
    "medtrinity_25m": {
        "repo_id": "UCSC-VLAA/MedTrinity-25M",
        "subset": "25M_demo",
        "split": "train",
        "max_samples": 10000
    }
}

# AIREADI 数据下载URL
AIREADI_DATA_URL = "https://docs.aireadi.org/docs/1/dataset/clinical-data/clinical-lab-tests/"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_path: str = SOURCE_MODEL_PATH
    output_dir: str = OUTPUT_MODEL_DIR
    
    # LoRA 配置
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 训练参数
    num_train_steps: int = 10000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_seq_length: int = 2048
    
    # 惩罚机制配置
    penalty_coefficient: float = 0.1  # 错误回答惩罚系数
    empathy_reward: float = 0.05      # 同理心表达奖励
    
    # 保存配置
    save_steps: int = 500
    eval_steps: int = 200
    logging_steps: int = 50
    
    # 设备配置
    use_gpu: bool = True  # 在Mac上使用GPU (Metal)
    use_cpu: bool = True  # 同时使用CPU进行部分计算
    mixed_precision: bool = True


@dataclass  
class DataConfig:
    """数据配置"""
    raw_data_dir: str = RAW_DATA_DIR
    processed_data_dir: str = PROCESSED_DATA_DIR
    
    # 图像配置
    image_size: int = 384
    max_image_tokens: int = 1024
    
    # 文本配置
    max_text_length: int = 1024
    
    # 数据增强
    use_augmentation: bool = True
    augmentation_prob: float = 0.3


# 人情味回复模板
EMPATHY_TEMPLATES = [
    "我理解您的担忧，让我来帮您分析一下这个情况。",
    "感谢您的信任，我会尽我所能为您提供准确的信息。",
    "我能感受到您对此的关切，这是很正常的反应。",
    "请不要太担心，让我们一起来看看这个结果。",
    "我很高兴能为您解答这个问题，让我详细说明一下。",
    "您提出了一个很好的问题，这显示您对健康的重视。",
    "我理解这可能让您感到焦虑，但请放心，我会认真为您分析。",
    "感谢您分享这些信息，让我来为您提供专业的见解。"
]

# 医疗术语解释风格指导
STYLE_GUIDELINES = """
回复风格指南：
1. 使用温和、专业的语气
2. 先表达理解和关心，再提供专业分析
3. 用通俗易懂的语言解释医学术语
4. 在适当时给予鼓励和支持
5. 提醒患者如有严重症状应及时就医
6. 不要做出绝对的诊断，但可以提供参考意见
7. 尊重患者的感受和担忧
"""

# 惩罚词汇列表（需要避免的表达）
PENALTY_WORDS = [
    "肯定是", "一定是", "必须", "绝对",  # 过于武断
    "不用担心", "没什么大不了",  # 过于轻视
    "你应该", "你必须",  # 命令式语气
    "这很简单", "很容易",  # 轻视问题
]

# 奖励词汇列表（鼓励使用的表达）
REWARD_WORDS = [
    "我理解", "我能感受到", "感谢您",  # 共情表达
    "让我来", "我来为您", "我会帮助您",  # 主动帮助
    "可能", "建议", "参考",  # 谨慎表达
    "如果", "通常来说", "一般情况下",  # 条件表达
]


def create_directories():
    """创建必要的目录"""
    dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        CHECKPOINT_DIR, LOG_DIR, OUTPUT_MODEL_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ 目录结构已创建")


if __name__ == "__main__":
    create_directories()
    print(f"📁 基础目录: {BASE_DIR}")
    print(f"📦 数据目录: {DATA_DIR}")
    print(f"🤖 模型路径: {SOURCE_MODEL_PATH}")

