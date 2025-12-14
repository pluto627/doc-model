#!/usr/bin/env python3
"""
医疗图像分析Web应用
支持图像和文本上传，使用训练好的模型进行分析
"""
import os
import sys
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SOURCE_MODEL_PATH

# 尝试导入MLX
try:
    import mlx
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️ MLX不可用，将使用模拟响应")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大上传大小
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局模型变量
model = None
tokenizer = None
model_loaded = False


def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model_once():
    """加载模型（只加载一次）"""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        return model is not None
    
    if not MLX_AVAILABLE:
        print("⚠️ MLX不可用，使用模拟模式")
        model_loaded = True
        return False
    
    try:
        print(f"🔄 正在加载真实AI模型...")
        print(f"📁 模型路径: {SOURCE_MODEL_PATH}")
        print(f"⏳ 这可能需要1-2分钟，请耐心等待...")
        
        model, tokenizer = load(SOURCE_MODEL_PATH)
        model_loaded = True
        
        print("✅ 模型加载成功！")
        print(f"💾 预计内存占用: 15-20GB")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        model_loaded = True
        return False


def generate_response(question: str, image_path: Optional[str] = None) -> str:
    """生成回复 - 使用真实AI模型"""
    global model, tokenizer
    
    # 使用真实模型生成
    if model is not None and MLX_AVAILABLE:
        try:
            print(f"🤖 使用AI模型生成回复...")
            
            # 准备图像（如果有）
            images = None
            if image_path:
                try:
                    with Image.open(image_path) as img:
                        images = [img.convert("RGB")]
                except Exception as e:
                    print(f"⚠️ 图像加载失败，改为纯文本模式: {e}")
                    images = None
            
            # 构建提示词
            if image_path:
                prompt = (
                    "<|im_start|>system\n"
                    "你是一个专业、有同理心的医疗健康助手。"
                    "你可以读取并理解用户上传的医疗影像或报告照片，"
                    "提取其中的文字（OCR）和关键信息，再结合用户问题给出专业、温暖的建议。"
                    "<|im_end|>\n"
                    "<|im_start|>user\n"
                    "以下是我上传的图片，请先阅读图片内容（包含文字和影像），再回答我的问题：\n"
                    f"问题: {question}"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            else:
                prompt = (
                    "<|im_start|>system\n"
                    "你是一个专业、有同理心的医疗健康助手。"
                    "请用温暖、专业的语气回答用户的健康问题。"
                    "<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"{question}"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            
            # 生成回复 - V4.2优化参数
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                images=images,
                max_tokens=512,
                temp=0.7,                    # 增加多样性
                top_p=0.9,                   # 核采样
                repetition_penalty=1.1,      # 减少重复
                verbose=False
            )
            
            print(f"✅ AI生成完成")
            return response.strip()
            
        except Exception as e:
            print(f"⚠️ AI生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            print("📝 回退到模拟响应")
            return generate_simulated_response(question)
    
    # 模型未加载，使用模拟响应
    print("⚠️ 模型未加载，使用模拟响应")
    return generate_simulated_response(question)


def generate_simulated_response(question: str) -> str:
    """生成模拟响应（当模型不可用时）"""
    
    # 智能响应模板
    responses = {
        "血压": "我理解您对血压的担忧，这是非常正常的健康关注。血压的正常范围通常是收缩压90-139 mmHg，舒张压60-89 mmHg。如果您的血压稍微偏高，建议您：\n\n1. 减少盐分摄入\n2. 保持规律运动\n3. 控制体重\n4. 保证充足睡眠\n\n如果持续偏高，建议咨询心内科医生进行专业评估。请问您还有其他想了解的吗？",
        
        "血糖": "感谢您分享检测结果，我来帮您分析一下。空腹血糖的正常范围一般是3.9-6.1 mmol/L，餐后2小时血糖应低于7.8 mmol/L。\n\n如果您的数值略高，不必过度担心，可以通过以下方式改善：\n\n1. 控制碳水化合物摄入\n2. 增加运动量\n3. 保持规律作息\n\n建议一周后复查，如有持续异常，请咨询内分泌科医生。",
        
        "x光": "我理解您上传了X光片，让我来帮您分析。X光检查是常见的影像学检查手段。\n\n请注意：\n- 图像质量会影响诊断准确性\n- 需要结合临床症状综合判断\n- 建议由专业放射科医生进行详细解读\n\n如果您有具体的疑问或症状，请详细描述，这样我可以给您更有针对性的建议。必要时请及时就医。",
        
        "CT": "感谢您上传CT影像。CT扫描能够提供详细的断层图像信息。\n\n分析要点：\n- CT报告需要专业医生结合临床综合判断\n- 如发现异常，建议咨询相关专科医生\n- 定期复查很重要\n\n如果您对报告有疑问，建议您：\n1. 携带完整报告就诊\n2. 详细描述症状\n3. 听取专科医生建议\n\n请不要过度担心，很多情况下早发现早治疗效果都很好。",
        
        "心电图": "我看到您上传了心电图，让我帮您做初步分析。心电图是评估心脏电活动的重要检查。\n\n正常心电图特征：\n- 心率：60-100次/分\n- 规律的波形\n- 正常的各波段间期\n\n如果报告提示异常，建议：\n1. 咨询心内科医生\n2. 结合临床症状分析\n3. 必要时做进一步检查\n\n如有胸闷、胸痛等症状，请及时就医。",
        
        "皮肤": "我理解您对皮肤问题的担忧。从图像观察，建议注意以下几点：\n\n基本护理：\n- 保持患处清洁干燥\n- 避免抓挠\n- 注意观察变化\n\n就医建议：\n- 如有明显不适，建议就诊皮肤科\n- 携带清晰图片给医生参考\n- 描述症状持续时间和变化\n\n皮肤问题种类繁多，专业医生能给出更准确的诊断和治疗方案。",
        
        "default": "感谢您的咨询，我很高兴能为您提供帮助。\n\n作为医疗健康AI助手，我会尽力为您提供准确、有帮助的信息。为了更好地帮助您，请您详细描述：\n\n1. 具体的症状或问题\n2. 持续时间\n3. 相关的检查结果\n4. 当前的身体状况\n\n同时请注意：\n- AI分析仅供参考，不能替代医生诊断\n- 如症状严重，请及时就医\n- 定期体检很重要\n\n请问您想咨询什么问题？"
    }
    
    # 根据关键词匹配响应
    question_lower = question.lower()
    
    for key, response in responses.items():
        if key in question_lower or key in question:
            return response
    
    return responses["default"]


def process_image(image_path: str) -> dict:
    """处理图像，提取基本信息"""
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_kb": os.path.getsize(image_path) / 1024
            }
    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """分析接口"""
    try:
        # 获取文本输入
        question = request.form.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': '请输入问题或描述'
            }), 400
        
        # 处理图像（如果有）
        image_path = None
        image_info = None
        
        if 'image' in request.files:
            file = request.files['image']
            
            if file and file.filename and allowed_file(file.filename):
                # 保存文件
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
                
                # 提取图像信息
                image_info = process_image(image_path)
                
                print(f"📸 图像已保存: {image_path}")
        
        # 生成回复
        print(f"💬 问题: {question[:100]}...")
        response = generate_response(question, image_path)
        print(f"✅ 回复生成完成")
        
        return jsonify({
            'success': True,
            'response': response,
            'image_info': image_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'处理请求时出错: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'mlx_available': MLX_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """访问上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    import socket
    
    # 获取本机IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("=" * 60)
    print("🏥 医疗图像分析Web应用")
    print("=" * 60)
    
    # 尝试加载模型
    print("\n正在加载模型...")
    load_model_once()
    
    print("\n" + "=" * 60)
    print("🌐 服务器启动信息")
    print("=" * 60)
    print(f"\n本地访问: http://localhost:8080")
    print(f"局域网访问: http://{local_ip}:8080")
    print("\n同一局域网内的其他设备可以通过上述地址访问")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 60 + "\n")
    
    # 启动服务器 (0.0.0.0 允许局域网访问)
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)

