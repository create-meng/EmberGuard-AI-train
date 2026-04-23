"""数字孪生演示系统 - Flask 后端"""
from flask import Flask, render_template, Response, request, jsonify
import os
import sys
import json
import threading
from datetime import datetime
import warnings
import time
import cv2
import numpy as np


_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '../..'))


def _abs_repo_path(p: str) -> str:
    """Resolve a path relative to repo root when a relative path is given."""
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(_REPO_ROOT, p))


ALARM_LOG_PATH = os.path.abspath(os.path.join(_HERE, 'alarm_logs.json'))
ALARM_TREND_PATH = os.path.abspath(os.path.join(_HERE, 'alarm_trend.json'))
_JSON_FILE_LOCK = threading.Lock()
ALARM_LOG_MAX_ITEMS = 20
ALARM_TREND_MAX_POINTS = 120

# 静默警告
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# 设置静默模式（隐藏所有初始化信息）
os.environ['SILENT_MODE'] = '1'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入管理器
from config_manager import ConfigManager
from detection_engine import DetectionEngine
from sensor_manager import SensorManager

# ===== Demo-only 后端 =====
# - 不使用 Socket.IO
# - 不注册 APIRouter
# - 不做楼层/建筑切换
# - 只提供 Demo 页面 + MJPEG + SSE

# 创建Flask应用
app = Flask(__name__, 
            static_folder='../static',
            template_folder='../frontend')

# 初始化（仅保留必要组件）
config_manager = ConfigManager()
sensor_manager = SensorManager(socketio=None, app=app)

# 获取模型路径
config = config_manager.get_config()
models = config.get('models', {})
yolo_path = _abs_repo_path(models.get('yolo', 'runs/detect/train2/weights/best.pt'))
lstm_path = _abs_repo_path(models.get('lstm', 'models/lstm/best.pt'))

# 是否显示技术字段（即如何得到最终告警）
SHOW_TECH_DETAILS_DEFAULT = False

EXPERIMENT_PROFILE = 'yolo_lstm_denoise_fusion'

_PROFILE_MAP = {
    'yolo': {'use_lstm': False, 'feature_denoise': False, 'frame_denoise': False, 'fusion': False},
    'yolo_lstm': {'use_lstm': True, 'feature_denoise': False, 'frame_denoise': False, 'fusion': False},
    'yolo_lstm_denoise': {'use_lstm': True, 'feature_denoise': True, 'frame_denoise': True, 'fusion': False},
    'yolo_lstm_fusion': {'use_lstm': True, 'feature_denoise': False, 'frame_denoise': False, 'fusion': True},
    'yolo_lstm_denoise_fusion': {'use_lstm': True, 'feature_denoise': True, 'frame_denoise': True, 'fusion': True},
}

_p = _PROFILE_MAP.get(EXPERIMENT_PROFILE) or _PROFILE_MAP['yolo_lstm_denoise_fusion']

# 初始化多个独立的检测引擎（每个摄像头一个）
detection_engines = {
    'demo_cam_001': DetectionEngine(yolo_path, lstm_path, use_lstm=_p['use_lstm'], enable_feature_denoise=_p['feature_denoise'], enable_frame_denoise=_p['frame_denoise'], enable_fusion=_p['fusion'], experiment_profile=EXPERIMENT_PROFILE),
    'demo_cam_002': DetectionEngine(yolo_path, lstm_path, use_lstm=_p['use_lstm'], enable_feature_denoise=_p['feature_denoise'], enable_frame_denoise=_p['frame_denoise'], enable_fusion=_p['fusion'], experiment_profile=EXPERIMENT_PROFILE),
    'demo_cam_003': DetectionEngine(yolo_path, lstm_path, use_lstm=_p['use_lstm'], enable_feature_denoise=_p['feature_denoise'], enable_frame_denoise=_p['frame_denoise'], enable_fusion=_p['fusion'], experiment_profile=EXPERIMENT_PROFILE),
}


def _start_demo_devices():
    # 多路演示：3个独立的视频源 + 单个传感器
    demo_video = _abs_repo_path(os.path.join('datasets', 'fire_videos_organized', 'fire', 'archive_fire2.mp4'))
    # 启动3个独立的摄像头
    camera_configs = [
        {'id': 'demo_cam_001', 'name': '摄像头01', 'point_id': 'CAM-01'},
        {'id': 'demo_cam_002', 'name': '摄像头02', 'point_id': 'CAM-02'},
        {'id': 'demo_cam_003', 'name': '摄像头03', 'point_id': 'CAM-03'},
    ]

    for cfg in camera_configs:
        engine = detection_engines.get(cfg['id'])
        if engine and os.path.exists(demo_video):
            engine.start(demo_video, name=cfg['name'], camera_id=cfg['id'])
            print(f"✓ {cfg['name']} ({cfg['point_id']}) 已启动")

    # 单个传感器（演示模式：始终模拟）
    sensor_manager.register_sensor(
        sensor_id='sensor_temp_001',
        sensor_type='temperature_sensor',
        threshold=60,
        name='温度传感器',
        unit='°C'
    )

    sensor_manager.register_sensor(
        sensor_id='sensor_hum_001',
        sensor_type='humidity_sensor',
        threshold=85,
        name='湿度传感器',
        unit='%'
    )
    sensor_manager.start_simulation()


_start_demo_devices()


def _read_alarm_logs_unlocked():
    if not os.path.exists(ALARM_LOG_PATH):
        return []
    with open(ALARM_LOG_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def _read_alarm_logs():
    try:
        with _JSON_FILE_LOCK:
            return _read_alarm_logs_unlocked()
    except Exception:
        return []


def _write_alarm_logs(items):
    try:
        if not isinstance(items, list):
            items = []
        tmp = ALARM_LOG_PATH + '.tmp'
        with _JSON_FILE_LOCK:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            try:
                os.replace(tmp, ALARM_LOG_PATH)
            except PermissionError:
                with open(ALARM_LOG_PATH, 'w', encoding='utf-8') as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        return True
    except Exception:
        return False


def _read_alarm_trend_unlocked():
    if not os.path.exists(ALARM_TREND_PATH):
        return []
    with open(ALARM_TREND_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    # 只保留数值点
    result = []
    for x in data:
        try:
            v = float(x)
            if v != v:
                continue
            result.append(v)
        except Exception:
            continue
    return result


def _read_alarm_trend():
    try:
        with _JSON_FILE_LOCK:
            return _read_alarm_trend_unlocked()
    except Exception:
        return []


def _write_alarm_trend(points):
    try:
        if not isinstance(points, list):
            points = []
        tmp = ALARM_TREND_PATH + '.tmp'
        with _JSON_FILE_LOCK:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(points, f, ensure_ascii=False, indent=2)
            try:
                os.replace(tmp, ALARM_TREND_PATH)
            except PermissionError:
                with open(ALARM_TREND_PATH, 'w', encoding='utf-8') as f:
                    json.dump(points, f, ensure_ascii=False, indent=2)
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        return True
    except Exception:
        return False


@app.route('/demo/alarm_trend', methods=['GET', 'POST', 'DELETE'])
def demo_alarm_trend():
    if request.method == 'GET':
        return jsonify({'points': _read_alarm_trend()})

    if request.method == 'DELETE':
        _write_alarm_trend([])
        return jsonify({'ok': True, 'points': []})

    payload = request.get_json(silent=True) or {}
    points = payload.get('points')
    if not isinstance(points, list):
        return jsonify({'ok': False, 'error': 'points must be list'}), 400

    mode = payload.get('mode')
    if mode not in (None, 'append', 'replace'):
        return jsonify({'ok': False, 'error': 'mode must be append|replace'}), 400

    # 限制长度，避免无限增长
    with _JSON_FILE_LOCK:
        merged = [] if mode == 'replace' else _read_alarm_trend_unlocked()
        for x in points:
            try:
                v = float(x)
                if v != v:
                    continue
                merged.append(v)
            except Exception:
                continue
        merged = merged[-ALARM_TREND_MAX_POINTS:]
        tmp = ALARM_TREND_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        try:
            os.replace(tmp, ALARM_TREND_PATH)
        except PermissionError:
            with open(ALARM_TREND_PATH, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            try:
                os.remove(tmp)
            except Exception:
                pass
    return jsonify({'ok': True, 'points': merged, 'mode': (mode or 'append')})


@app.route('/demo/alarm_logs', methods=['GET', 'POST', 'DELETE'])
def demo_alarm_logs():
    if request.method == 'GET':
        return jsonify({'items': _read_alarm_logs()})

    if request.method == 'DELETE':
        _write_alarm_logs([])
        return jsonify({'ok': True, 'items': []})

    payload = request.get_json(silent=True) or {}
    level = payload.get('level')
    camera_id = payload.get('cameraId')
    now = datetime.now()
    ts = payload.get('ts') or now.strftime('%H:%M:%S')
    day = payload.get('date') or now.strftime('%Y-%m-%d')

    if level not in ('fire', 'smoke'):
        return jsonify({'ok': False, 'error': 'invalid level'}), 400

    item = {
        'date': str(day),
        'ts': ts,
        'cameraId': str(camera_id) if camera_id else '-',
        'level': level,
    }

    with _JSON_FILE_LOCK:
        items = _read_alarm_logs_unlocked()
        items.insert(0, item)
        items = items[:ALARM_LOG_MAX_ITEMS]
        tmp = ALARM_LOG_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        try:
            os.replace(tmp, ALARM_LOG_PATH)
        except PermissionError:
            with open(ALARM_LOG_PATH, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            try:
                os.remove(tmp)
            except Exception:
                pass
    return jsonify({'ok': True, 'items': items})


@app.route('/')
def demo_index():
    return render_template('demo.html', experiment_profile=EXPERIMENT_PROFILE, show_tech_details_default=SHOW_TECH_DETAILS_DEFAULT)


@app.route('/demo/events')
def demo_events():
    """Demo 结果流：SSE 低频推送所有摄像头的推理结果 + 传感器快照，避免 Socket 高频更新卡顿。"""
    def gen():
        while True:
            try:
                payload = {
                    'ts': datetime.now().strftime('%H:%M:%S'),
                    'cameras': [],  # 改为数组，包含所有摄像头数据
                    'sensors': None
                }

                # 获取所有摄像头的快照
                for camera_id, engine in detection_engines.items():
                    snapshot = engine.get_snapshot()
                    if snapshot:
                        payload['cameras'].append(snapshot)

                # 传感器快照（轻量）
                try:
                    payload['sensors'] = sensor_manager.get_all_sensors()
                except Exception:
                    payload['sensors'] = []

                data = json.dumps(payload, ensure_ascii=False)
                yield f"data: {data}\n\n"
                time.sleep(0.5)
            except GeneratorExit:
                return
            except Exception:
                time.sleep(0.5)

    return Response(gen(), mimetype='text/event-stream')


def _mjpeg_response(camera_id=None):
    # 根据camera_id选择对应的检测引擎
    engine = None
    if camera_id and camera_id in detection_engines:
        engine = detection_engines[camera_id]
    elif not camera_id:
        # 默认使用第一个摄像头
        engine = list(detection_engines.values())[0] if detection_engines else None

    def gen():
        if engine and hasattr(engine, 'add_stream_client'):
            try:
                engine.add_stream_client()
            except Exception:
                pass
        try:
            while True:
                try:
                    if engine:
                        jpeg = engine.get_latest_jpeg()
                        if jpeg:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                        else:
                            time.sleep(0.01)
                    else:
                        time.sleep(0.1)
                except GeneratorExit:
                    return
                except Exception:
                    time.sleep(0.1)
        finally:
            if engine and hasattr(engine, 'remove_stream_client'):
                try:
                    engine.remove_stream_client()
                except Exception:
                    pass

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/demo/stream')
def demo_stream():
    from flask import request
    camera_id = request.args.get('camera_id', None)
    # 兼容前端可能传入的CAM-01格式，转换为demo_cam_001
    if camera_id:
        # 尝试从CAM-01格式转换为demo_cam_001
        import re
        m = re.match(r'CAM-(\d+)', camera_id)
        if m:
            num = m.group(1)
            camera_id = f'demo_cam_{num.zfill(3)}'
    return _mjpeg_response(camera_id)

if __name__ == '__main__':
    # 检查模型状态（使用第一个引擎的状态）
    first_engine = list(detection_engines.values())[0] if detection_engines else None
    model_status = '✓' if first_engine and getattr(first_engine, 'pipeline_available', False) else '✗'
    cam_count = len(detection_engines)
    print(f"🚀 Demo 服务器启动 http://localhost:5000 | 摄像头数量: {cam_count} | 模型: {model_status}")
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
