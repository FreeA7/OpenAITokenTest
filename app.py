import logging
import time
import re
import datetime
import json

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from openai import OpenAI

# -----------------------
# Flask 应用及数据库配置
# -----------------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///api_calls.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -----------------------
# 日志配置（utf-8编码）
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PID %(process)d] [Thread %(thread)d] %(module)s:%(lineno)d %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8'
)
logger = logging.getLogger(__name__)

# -----------------------
# SQLAlchemy 数据模型
# -----------------------
class APICall(db.Model):
    __tablename__ = 'api_calls'
    uuid = db.Column(db.String(64), primary_key=True)
    messages = db.Column(db.Text, nullable=False)
    model = db.Column(db.String(64), nullable=False)
    response_format = db.Column(db.String(64), nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    reply = db.Column(db.Text, nullable=True)
    prompt_tokens = db.Column(db.Integer, nullable=True)
    completion_tokens = db.Column(db.Integer, nullable=True)
    total_tokens = db.Column(db.Integer, nullable=True)
    call_duration = db.Column(db.Float, nullable=False)
    error_flag = db.Column(db.Integer, nullable=False, default=0)
    call_time = db.Column(db.DateTime, nullable=False)
    request_ip = db.Column(db.String(64), nullable=False)

with app.app_context():
    db.create_all()

# -----------------------
# API 接口实现
# -----------------------
@app.route('/api/call', methods=['POST'])
def call_openai():
    data = request.get_json()
    if not data:
        return jsonify({'error': '无效的JSON数据'}), 400

    # 提取参数
    api_key = data.get('api_key')
    messages = data.get('messages')
    model_name = data.get('model')
    response_format = data.get('response_format', 'text')
    uuid = data.get('uuid')
    temperature = data.get('temperature', 1.0)

    logger.info(f"[{uuid}]********** Start **********")
    logging.info(
        f'[{uuid}]Get a chat request from ip: {request.remote_addr}, model: {model_name}, response_format: {response_format}, temperature: {temperature}')

    if not all([api_key, messages, model_name, response_format, uuid]):
        logger.error(f"[{uuid}]缺少必要参数")
        return jsonify({'error': '缺少必要参数'}), 400

    # 设置 OpenAI API 密钥
    client = OpenAI(api_key=api_key)

    # 记录开始时间
    start_time = time.time()
    try:
        if response_format == 'json':
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
        elif response_format == 'text':
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
        else:
            raise AttributeError('Error response_format')
    except Exception as e:
        logger.error(f"[{uuid}]调用 OpenAI API 失败: {str(e)}")
        return jsonify({'error': '调用 OpenAI API 失败', 'details': str(e)}), 500

    # 记录调用时长
    duration = time.time() - start_time

    # 解析 API 返回结果
    reply = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    # 使用正则表达式检查 reply 中换行符连续出现4个或以上
    error_flag = 1 if re.search(r'[\n\r]{4,}', reply) else 0

    # 使用 SQLAlchemy 保存调用记录（确保中文以 utf-8 编码存储）
    call_record = APICall(
        uuid=uuid,
        messages=json.dumps(messages, ensure_ascii=False),
        model=model_name,
        response_format=response_format,
        temperature=temperature,
        reply=reply,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        call_duration=duration,
        error_flag=error_flag,
        call_time=datetime.datetime.utcnow(),
        request_ip=request.remote_addr or 'unknown',
    )

    try:
        db.session.add(call_record)
        db.session.commit()
    except Exception as e:
        logger.error(f"[{uuid}]数据库错误: {str(e)}")
        db.session.rollback()
        return jsonify({'error': '数据库错误', 'details': str(e)}), 500

    # 打印日志
    logger.info(f"[{uuid}]API调用记录 IP: {request.remote_addr}, 时长: {duration}秒, error_flag: {error_flag}")
    logger.info(f"[{uuid}]Prompt_tokens: {prompt_tokens}")
    logger.info(f"[{uuid}]completion_tokens: {completion_tokens}")
    logger.info(f"[{uuid}]total_tokens: {total_tokens}")

    # 返回 OpenAI 的部分返回值及调用信息
    return jsonify({
        'reply': reply,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'call_duration': duration,
        'error_flag': error_flag,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
