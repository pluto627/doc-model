#!/usr/bin/env python3
"""
准备 V4.2 训练数据
重点改进：
1. 智能判断是否追问（情境化回复）
2. 回复多样性
3. 准确性（精准医学信息）
4. 自然的人情味
"""

import json
import random
from pathlib import Path
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()

# 输出目录
OUTPUT_DIR = Path("data_v42")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 改进的人情味模板（更自然，不机械）
# ============================================================

# 开头语（根据情境选择，不是每次都用）
NATURAL_OPENERS = [
    "",  # 有时候不需要开头语，直接说重点
    "好的，",
    "我来看看。",
    "明白了。",
    "了解。",
    "收到。",
]

# 结尾语（根据情境选择）
NATURAL_ENDINGS_COMPLETE = [
    # 信息完整时的结尾（不追问）
    "",
    "\n\n有其他问题随时问我~",
    "\n\n希望对您有帮助。",
    "\n\n祝您健康！",
    "\n\n如有变化及时告诉我。",
]

NATURAL_ENDINGS_NEED_INFO = [
    # 需要更多信息时的结尾（追问）
    "\n\n方便告诉我这些信息吗？",
    "\n\n您可以简单回复，我来帮您分析~",
    "\n\n这样我能给您更准确的建议。",
]

# ============================================================
# 精准医学知识库
# ============================================================

PRECISE_MEDICAL_KNOWLEDGE = [
    # 血压相关（精准数值）
    {
        "q": "血压多少算正常？",
        "a": """血压的正常范围：

📊 **分级标准**（单位：mmHg）
| 分类 | 收缩压 | 舒张压 |
|------|--------|--------|
| 正常 | <120 | <80 |
| 正常高值 | 120-139 | 80-89 |
| 1级高血压 | 140-159 | 90-99 |
| 2级高血压 | 160-179 | 100-109 |
| 3级高血压 | ≥180 | ≥110 |

💡 **注意事项**：
- 测量前休息5分钟
- 避免咖啡、运动后立即测量
- 建议多次测量取平均值"""
    },
    {
        "q": "高血压吃什么药？",
        "a": """常见降压药物分类：

💊 **五大类降压药**：

1️⃣ **ACEI类**（普利类）
   - 代表药：依那普利、贝那普利
   - 特点：保护心肾，适合糖尿病患者
   - 注意：可能引起干咳

2️⃣ **ARB类**（沙坦类）
   - 代表药：氯沙坦、缬沙坦
   - 特点：不引起干咳，耐受性好

3️⃣ **CCB类**（地平类）
   - 代表药：氨氯地平、硝苯地平
   - 特点：降压平稳，适合老年人

4️⃣ **利尿剂**
   - 代表药：氢氯噻嗪、吲达帕胺
   - 特点：价格便宜，常与其他药联用

5️⃣ **β受体阻滞剂**
   - 代表药：美托洛尔、比索洛尔
   - 特点：适合心率快的患者

⚠️ 具体用药需医生根据您的情况处方，切勿自行用药。"""
    },
    
    # 血糖相关（精准数值）
    {
        "q": "血糖正常值是多少？",
        "a": """血糖正常参考范围：

📊 **诊断标准**（单位：mmol/L）

| 指标 | 正常 | 糖尿病前期 | 糖尿病 |
|------|------|-----------|--------|
| 空腹血糖 | <6.1 | 6.1-6.9 | ≥7.0 |
| 餐后2h | <7.8 | 7.8-11.0 | ≥11.1 |
| 随机血糖 | - | - | ≥11.1+症状 |

📌 **糖化血红蛋白（HbA1c）**：
- 正常：<6.0%
- 糖尿病前期：6.0-6.4%
- 糖尿病：≥6.5%

💡 **测量要点**：
- 空腹：禁食8-10小时
- 餐后2h：从第一口饭开始计时"""
    },
    {
        "q": "糖尿病怎么控制饮食？",
        "a": """糖尿病饮食控制要点：

🍚 **主食控制**：
- 每餐主食：1-2两（50-100g）
- 选择粗粮：糙米、燕麦、荞麦
- 避免：白粥、白面包、糯米

🥬 **蔬菜优先**：
- 每天500g以上蔬菜
- 优选：叶菜类、瓜类
- 限制：土豆、山药（算主食）

🍖 **蛋白质适量**：
- 瘦肉、鱼虾、豆腐
- 每天1个鸡蛋
- 牛奶250ml

🍎 **水果限量**：
- 选低GI水果：苹果、柚子、草莓
- 每天不超过200g
- 放在两餐之间吃

⏰ **进餐规律**：
- 定时定量
- 细嚼慢咽
- 先菜后饭"""
    },
    
    # 肝功能相关
    {
        "q": "肝功能检查指标怎么看？",
        "a": """肝功能主要指标解读：

📊 **转氨酶**：
| 指标 | 正常范围 | 临床意义 |
|------|---------|---------|
| ALT | 0-40 U/L | 肝细胞损伤敏感指标 |
| AST | 0-40 U/L | 肝脏/心肌损伤 |

📊 **胆红素**：
| 指标 | 正常范围 | 升高提示 |
|------|---------|---------|
| 总胆红素 | 3.4-17.1 μmol/L | 黄疸 |
| 直接胆红素 | 0-6.8 μmol/L | 胆道梗阻 |

📊 **蛋白质**：
| 指标 | 正常范围 | 降低提示 |
|------|---------|---------|
| 白蛋白 | 35-55 g/L | 肝功能受损 |
| 球蛋白 | 20-30 g/L | 免疫相关 |

💡 **常见升高原因**：
- 饮酒、药物、熬夜
- 脂肪肝、病毒性肝炎
- 轻度升高可观察，持续升高需就医"""
    },
    
    # 血常规相关
    {
        "q": "血常规怎么看？",
        "a": """血常规主要指标解读：

🔴 **红细胞系列**：
| 指标 | 男性正常值 | 女性正常值 |
|------|-----------|-----------|
| 红细胞(RBC) | 4.0-5.5×10¹²/L | 3.5-5.0×10¹²/L |
| 血红蛋白(Hb) | 120-160 g/L | 110-150 g/L |
| 红细胞压积 | 40-50% | 37-45% |

⚪ **白细胞系列**：
| 指标 | 正常范围 | 临床意义 |
|------|---------|---------|
| 白细胞(WBC) | 4-10×10⁹/L | 升高提示感染 |
| 中性粒细胞 | 50-70% | 细菌感染升高 |
| 淋巴细胞 | 20-40% | 病毒感染升高 |

🟡 **血小板**：
- 正常：100-300×10⁹/L
- 降低：出血风险增加
- 升高：血栓风险增加"""
    },
    
    # 用药时间
    {
        "q": "降压药什么时候吃最好？",
        "a": """降压药服用时间建议：

⏰ **长效降压药**（每日1次）：
- **推荐**：早晨起床后服用
- 原因：血压晨峰现象，早晨血压最高
- 代表药：氨氯地平、缬沙坦

⏰ **短效降压药**（每日2-3次）：
- 按说明书规定间隔服用
- 保持血药浓度稳定

⏰ **利尿类降压药**：
- **必须**早晨服用
- 原因：避免夜间频繁起夜

💡 **特殊情况**：
- 夜间高血压：可睡前服用
- 具体方案遵医嘱

⚠️ **注意事项**：
- 每天固定时间服药
- 不要漏服或自行停药
- 定期监测血压"""
    },
    {
        "q": "感冒药什么时候吃？",
        "a": """感冒药服用时间指南：

💊 **解热镇痛药**（对乙酰氨基酚、布洛芬）：
- **建议**：饭后30分钟服用
- 原因：减少胃肠刺激
- 间隔：至少4-6小时

💊 **抗组胺药**（氯雷他定、西替利嗪）：
- 嗜睡型（扑尔敏）：睡前服用
- 非嗜睡型（氯雷他定）：早晨服用

💊 **止咳药**：
- 白天咳嗽：选不嗜睡的
- 夜间咳嗽：睡前服用含镇咳成分的

💊 **复方感冒药**：
- 按说明书服用
- 通常每4-6小时一次
- 注意不要与同类药物叠加

⚠️ **重要提醒**：
- 多喝水、多休息
- 症状超过1周未好转需就医
- 发热超过3天需就医"""
    },
]

# ============================================================
# 情境化回复样本（智能判断是否追问）
# ============================================================

CONTEXTUAL_SAMPLES = [
    # ========== 类型A：信息充足 → 直接回答 ==========
    {
        "messages": [
            {"role": "user", "content": "我的血压是145/92，早上空腹测的，之前休息了5分钟，没吃降压药"},
            {"role": "assistant", "content": """好的，我来帮您分析这个血压数值。

📊 **数据解读**：
- 收缩压 145 mmHg（正常<140）
- 舒张压 92 mmHg（正常<90）
- 属于 **1级高血压**

您的测量条件很规范（空腹、休息后），数据比较可靠。

💡 **建议**：
1. 连续监测一周，每天同一时间测量
2. 记录数值，观察变化趋势
3. 减少盐分摄入（每日<6g）
4. 如持续偏高，建议心内科就诊

单次测量偏高不必太紧张，持续监测更重要~"""}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "血糖空腹6.8，餐后2小时9.2，这个情况怎么样？"},
            {"role": "assistant", "content": """您的血糖数值我来帮您分析：

📊 **数据对照**：
| 指标 | 您的数值 | 正常范围 | 判断 |
|------|---------|---------|------|
| 空腹血糖 | 6.8 | <6.1 | 偏高 |
| 餐后2h | 9.2 | <7.8 | 偏高 |

📌 **综合判断**：属于"糖尿病前期"（糖耐量异常）

这个阶段很关键，通过生活方式干预，很多人可以逆转！

✅ **建议措施**：
1. **饮食**：减少精制碳水，增加蔬菜和粗粮
2. **运动**：每周150分钟中等强度运动
3. **体重**：如超重，减重5-7%效果显著
4. **复查**：1-2个月后复查血糖和糖化血红蛋白

现在干预效果最好，加油！"""}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "肝功能检查ALT 85，AST 62，其他都正常"},
            {"role": "assistant", "content": """我来帮您分析肝功能结果：

📊 **数据解读**：
| 指标 | 您的数值 | 正常范围 | 状态 |
|------|---------|---------|------|
| ALT | 85 U/L | <40 | 偏高约2倍 |
| AST | 62 U/L | <40 | 偏高约1.5倍 |

ALT升高为主，提示肝细胞有一定损伤。

🔍 **可能原因**：
- 近期饮酒、熬夜
- 服用某些药物（如止痛药、降脂药）
- 脂肪肝
- 病毒性肝炎（需进一步检查排除）

✅ **建议**：
1. 近两周避免饮酒
2. 规律作息，避免熬夜
3. 清淡饮食
4. 3-4周后复查肝功能
5. 如有乙肝疫苗接种史，一般可排除乙肝

轻度升高调整后多数能恢复，不必过度担心。"""}
        ]
    },
    
    # ========== 类型B：信息不足 → 适当追问 ==========
    {
        "messages": [
            {"role": "user", "content": "我血压有点高"},
            {"role": "assistant", "content": """血压偏高的情况因人而异，需要了解更多才能给您准确建议。

方便告诉我：
1. 具体数值是多少？（比如145/90）
2. 什么时候测的？（早上/晚上，饭前/饭后）

这样我能帮您判断是否需要重视~"""}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "我血糖高"},
            {"role": "assistant", "content": """血糖偏高需要看具体数值才能判断。请问：

1. 空腹还是餐后测的？
2. 具体数值是多少？

告诉我这些，我来帮您分析~"""}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "帮我看看这个报告"},
            {"role": "assistant", "content": """好的，请问：

1. 这是什么检查的报告？（血常规/肝功能/肾功能等）
2. 有哪些指标异常吗？

您可以把异常的指标告诉我，或者描述一下报告内容~"""}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "我最近身体不太舒服"},
            {"role": "assistant", "content": """了解，方便描述一下具体是哪里不舒服吗？

比如：
- 是哪个部位？
- 什么样的感觉？（疼痛/酸胀/乏力等）
- 持续多久了？

这样我能更好地帮您分析~"""}
        ]
    },
    
    # ========== 类型C：纯知识问答 → 直接回答 ==========
    {
        "messages": [
            {"role": "user", "content": "高血压和低血压的标准是什么？"},
            {"role": "assistant", "content": """血压判断标准：

📊 **高血压**（单位：mmHg）：
- 收缩压 ≥140 或 舒张压 ≥90
- 需要多次测量确认

📊 **低血压**：
- 收缩压 <90 或 舒张压 <60
- 伴有头晕、乏力等症状

📊 **正常血压**：
- 收缩压 90-139
- 舒张压 60-89

💡 理想血压是 120/80 左右。"""}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "什么是糖化血红蛋白？"},
            {"role": "assistant", "content": """糖化血红蛋白（HbA1c）是反映血糖控制的重要指标：

📌 **原理**：
血液中的葡萄糖与红细胞中的血红蛋白结合，形成糖化血红蛋白。

📌 **特点**：
- 反映过去2-3个月的平均血糖水平
- 不受单次饮食、运动影响
- 比空腹血糖更稳定可靠

📊 **参考范围**：
| 数值 | 判断 |
|------|------|
| <6.0% | 正常 |
| 6.0-6.4% | 糖尿病前期 |
| ≥6.5% | 糖尿病诊断标准 |

💡 糖尿病患者控制目标一般是 <7.0%。"""}
        ]
    },
    
    # ========== 类型D：多轮对话 ==========
    {
        "messages": [
            {"role": "user", "content": "我血压偏高怎么办"},
            {"role": "assistant", "content": """血压偏高需要根据具体数值来判断处理方式。方便告诉我您的血压数值吗？"""},
            {"role": "user", "content": "152/95"},
            {"role": "assistant", "content": """152/95 mmHg 属于1级高血压。

✅ **建议**：
1. **生活调整**（先尝试2-4周）：
   - 限盐：每日<6g
   - 运动：每周150分钟
   - 限酒、戒烟
   - 控制体重

2. **监测**：
   - 每天定时测量记录
   - 观察生活调整后的变化

3. **就医指征**：
   - 调整后仍≥140/90
   - 有头痛、胸闷等症状
   - 有糖尿病、肾病等合并症

如果持续偏高，建议心内科就诊评估是否需要药物治疗。"""}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "最近总是头晕"},
            {"role": "assistant", "content": """头晕的原因比较多，需要了解更多信息。请问：

1. 头晕时有天旋地转的感觉吗？还是昏沉沉的？
2. 什么时候容易发作？（起床时/转头时/持续性）
3. 有其他症状吗？（恶心、耳鸣、心慌等）"""},
            {"role": "user", "content": "起床的时候会晕，躺一会就好了，没有其他症状"},
            {"role": "assistant", "content": """根据您描述的"起床时头晕，躺下缓解"，这很可能是 **体位性低血压**。

📌 **特点**：
- 从躺到站时血压突然下降
- 导致大脑短暂供血不足
- 躺下后很快恢复

🔍 **常见原因**：
- 起床太快
- 服用某些降压药
- 脱水
- 自主神经功能问题

✅ **建议**：
1. **起床"三慢"**：醒后躺30秒→坐30秒→站30秒
2. 多喝水，避免脱水
3. 测量卧位和立位血压（躺着测一次，站起来后测一次）
4. 如果频繁发作，建议就诊排查原因

试试慢起床的方法，应该会有改善~"""}
        ]
    },
]

# ============================================================
# 图像分析相关（情境化）
# ============================================================

IMAGE_ANALYSIS_SAMPLES = [
    # 信息充足的图像分析
    {
        "messages": [
            {"role": "user", "content": "这是我的胸片报告，结论写着：双肺纹理增多，心影不大，未见明显实质性病变"},
            {"role": "assistant", "content": """我来帮您解读这个胸片报告：

📋 **报告解读**：

1️⃣ **双肺纹理增多**
- 这是最常见的描述之一
- 可能原因：吸烟、感冒后、慢性支气管炎
- 很多情况下是正常变异

2️⃣ **心影不大**
- 说明心脏大小正常，排除心脏增大

3️⃣ **未见明显实质性病变**
- 没有发现肺炎、肿块等明显异常
- 这是好消息

📌 **总结**：
这个报告基本正常，没有发现需要紧急处理的问题。"肺纹理增多"描述很常见，单独存在一般不需要特殊处理。

如果您有咳嗽等症状，对症处理即可；没有症状的话不用担心~"""}
        ]
    },
    # 需要更多信息的图像分析
    {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": "path/to/image.jpg"},
                {"type": "text", "text": "帮我看看这个"}
            ]},
            {"role": "assistant", "content": """我看到您上传了一张图片。为了更好地帮您分析：

1. 这是什么类型的检查？（X光/CT/B超/化验单等）
2. 您有什么不舒服或担心的问题吗？

告诉我这些，我能给您更有针对性的解读~"""}
        ]
    },
]

# ============================================================
# 身份认知（更自然）
# ============================================================

IDENTITY_SAMPLES = [
    {"q": "你是谁？", "a": "我是XunDoc，您的健康助手~有什么健康问题可以帮您？"},
    {"q": "你叫什么名字？", "a": "我叫XunDoc，您可以叫我小寻~是一个医疗健康AI助手。"},
    {"q": "你能做什么？", "a": "我可以帮您：\n1. 解读体检报告和化验单\n2. 分析医学影像\n3. 解答健康问题\n4. 提供健康建议\n\n有什么想咨询的吗？"},
    {"q": "你好", "a": "您好！我是XunDoc，有什么可以帮您的？"},
    {"q": "早上好", "a": "早上好！希望您今天精神满满~有什么健康问题想咨询吗？"},
    {"q": "谢谢", "a": "不客气~能帮到您是我的荣幸。有问题随时找我！"},
    {"q": "Who are you?", "a": "I'm XunDoc, your health assistant. How can I help you today?"},
    {"q": "What can you do?", "a": "I can help with:\n1. Interpreting medical reports\n2. Analyzing medical images\n3. Answering health questions\n4. Providing health advice\n\nWhat would you like to know?"},
]


def load_existing_data():
    """加载现有的训练数据"""
    console.print("\n[cyan]📥 加载现有数据...[/cyan]")
    
    samples = []
    
    # 加载 v4.1 的数据作为基础
    v41_train = Path("data_v41/train.jsonl")
    if v41_train.exists():
        with open(v41_train, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    samples.append(json.loads(line))
                except:
                    continue
        console.print(f"  ✅ 加载了 {len(samples)} 条 v4.1 数据")
    
    return samples


def create_precision_samples():
    """创建精准医学知识样本"""
    console.print("\n[cyan]📝 创建精准医学知识样本...[/cyan]")
    
    samples = []
    for item in PRECISE_MEDICAL_KNOWLEDGE:
        samples.append({
            "messages": [
                {"role": "user", "content": item["q"]},
                {"role": "assistant", "content": item["a"]}
            ]
        })
    
    # 重复增强
    samples = samples * 25  # 每条重复25次
    console.print(f"  ✅ 创建了 {len(samples)} 条精准医学样本")
    return samples


def create_contextual_samples():
    """创建情境化回复样本"""
    console.print("\n[cyan]📝 创建情境化回复样本...[/cyan]")
    
    samples = CONTEXTUAL_SAMPLES.copy()
    
    # 重复增强
    samples = samples * 30  # 每条重复30次
    random.shuffle(samples)
    
    console.print(f"  ✅ 创建了 {len(samples)} 条情境化样本")
    return samples


def create_identity_samples():
    """创建身份认知样本"""
    console.print("\n[cyan]📝 创建身份认知样本...[/cyan]")
    
    samples = []
    for item in IDENTITY_SAMPLES:
        samples.append({
            "messages": [
                {"role": "user", "content": item["q"]},
                {"role": "assistant", "content": item["a"]}
            ]
        })
    
    # 重复增强
    samples = samples * 40  # 每条重复40次
    console.print(f"  ✅ 创建了 {len(samples)} 条身份样本")
    return samples


def improve_existing_sample(sample):
    """改进现有样本的结尾"""
    try:
        messages = sample.get("messages", [])
        if len(messages) >= 2:
            assistant_msg = messages[-1].get("content", "")
            
            # 移除机械化的结尾
            mechanical_endings = [
                "如果您还有其他问题，我很乐意为您解答。",
                "如有其他疑问，请随时告诉我。",
                "希望这些信息对您有帮助。",
                "有任何问题都可以继续咨询，我随时都在！",
                "如有疑问，请随时告诉我，我会尽力帮您解答~",
                "有什么不明白的地方随时问我哦~",
            ]
            
            for ending in mechanical_endings:
                if assistant_msg.endswith(ending):
                    assistant_msg = assistant_msg[:-len(ending)].rstrip()
                    # 随机决定是否添加新结尾
                    if random.random() < 0.3:  # 30%概率添加简短结尾
                        new_ending = random.choice([
                            "",
                            "\n\n有问题随时问~",
                            "\n\n祝您健康！",
                        ])
                        assistant_msg = assistant_msg + new_ending
                    break
            
            messages[-1]["content"] = assistant_msg
            sample["messages"] = messages
    except:
        pass
    
    return sample


def main():
    console.print("\n" + "="*60)
    console.print("[bold cyan]🏥 准备 V4.2 训练数据[/bold cyan]")
    console.print("[bold]重点：智能判断+回复多样性+准确性+自然人情味[/bold]")
    console.print("="*60)
    
    all_samples = []
    
    # 1. 加载现有数据
    existing = load_existing_data()
    # 改进现有样本
    console.print("\n[cyan]🔧 改进现有样本...[/cyan]")
    improved = [improve_existing_sample(s) for s in track(existing, description="改进中...")]
    all_samples.extend(improved)
    
    # 2. 创建精准医学知识样本
    precision = create_precision_samples()
    all_samples.extend(precision)
    
    # 3. 创建情境化回复样本
    contextual = create_contextual_samples()
    all_samples.extend(contextual)
    
    # 4. 创建身份认知样本
    identity = create_identity_samples()
    all_samples.extend(identity)
    
    # 5. 添加图像分析样本
    image_samples = IMAGE_ANALYSIS_SAMPLES * 20
    all_samples.extend(image_samples)
    
    # 6. 去重
    console.print("\n[cyan]🔄 去重处理...[/cyan]")
    seen = set()
    unique_samples = []
    for s in all_samples:
        key = json.dumps(s, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)
    
    console.print(f"  去重前: {len(all_samples)} → 去重后: {len(unique_samples)}")
    
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
    console.print("[bold green]✅ V4.2 数据准备完成！[/bold green]")
    console.print("="*60)
    console.print(f"\n📊 数据统计：")
    console.print(f"  训练集: {len(train_data):,} 条")
    console.print(f"  验证集: {len(valid_data):,} 条")
    console.print(f"  测试集: {len(test_data):,} 条")
    console.print(f"  总计: {total:,} 条")
    console.print(f"\n📁 保存位置: {OUTPUT_DIR}/")
    console.print(f"\n💡 建议训练步数: 5000 步")


if __name__ == "__main__":
    main()


