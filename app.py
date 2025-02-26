import logging
import time
import re
import datetime
import json

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import openai

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
    response_format = data.get('response_format')
    uuid_val = data.get('uuid')
    temperature = data.get('temperature', 1.0)

    if not all([api_key, messages, model_name, response_format, uuid_val]):
        return jsonify({'error': '缺少必要参数'}), 400

    # 设置 OpenAI API 密钥
    openai.api_key = api_key

    # 记录开始时间
    start_time = time.time()
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
    except Exception as e:
        logger.error("调用 OpenAI API 失败: %s", str(e))
        return jsonify({'error': '调用 OpenAI API 失败', 'details': str(e)}), 500

    # 记录调用时长
    duration = time.time() - start_time

    # 解析 API 返回结果
    reply = response.choices[0].message.get('content', '')
    prompt_tokens = response.get('usage', {}).get('prompt_tokens', 0)
    completion_tokens = response.get('usage', {}).get('completion_tokens', 0)

    # 使用正则表达式检查 reply 中换行符连续出现4个或以上
    error_flag = 1 if re.search(r'[\n\r]{4,}', reply) else 0

    # 使用 SQLAlchemy 保存调用记录（确保中文以 utf-8 编码存储）
    call_record = APICall(
        uuid=uuid_val,
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
        logger.error("数据库错误: %s", str(e))
        db.session.rollback()
        return jsonify({'error': '数据库错误', 'details': str(e)}), 500

    # 打印日志
    logger.info("API调用记录 UUID %s, IP: %s, 时长: %.2f秒, error_flag: %d",
                uuid_val, request.remote_addr, duration, error_flag)

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
