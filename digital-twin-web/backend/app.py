"""
数字孪生古建筑火灾监控Web系统 - Flask后端主应用
"""
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import os
import sys
import json
from datetime import datetime
import warnings
import time
import cv2
import numpy as np
from pathlib import Path

# 静默警告
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# 设置静默模式（隐藏所有初始化信息）
os.environ['SILENT_MODE'] = '1'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 项目根目录（ultralytics-main），用于解析 demo_video 相对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

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

# 配置CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# 初始化（仅保留必要组件）
config_manager = ConfigManager()
sensor_manager = SensorManager(socketio=None, app=app)

# 获取模型路径
config = config_manager.get_config()
models = config.get('models', {})
yolo_path = models.get('yolo', 'runs/detect/train2/weights/best.pt')
lstm_path = models.get('lstm', 'models/lstm/best.pt')

# 初始化检测引擎
detection_engine = DetectionEngine(
    yolo_path,
    lstm_path,
    socketio=None,
    alert_manager=None,
    history_manager=None,
    video_recorder=None,
)


def _load_demo_building_config() -> dict:
    cfg_path = Path(__file__).parent.parent / 'buildings' / 'demo' / 'config.json'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _start_demo_devices():
    building_config = _load_demo_building_config()

    facilities = []
    if 'floors' in building_config:
        for floor in building_config.get('floors', []):
            facilities.extend(floor.get('facilities', []))
    else:
        facilities = building_config.get('facilities', [])

    # 注册传感器
    for facility in facilities:
        if facility.get('type') in ['temperature_sensor', 'humidity_sensor', 'smoke_detector']:
            sensor_manager.register_sensor(
                sensor_id=facility['id'],
                sensor_type=facility['type'],
                threshold=facility.get('threshold'),
                name=facility.get('name'),
                unit=facility.get('unit')
            )

    # 启动摄像头（强制 demo 视频源）
    camera_configs = [f for f in facilities if f.get('type') == 'camera']
    if camera_configs:
        detection_engine.start_all_cameras(camera_configs, use_demo_video=True)

    # 启动传感器模拟（Demo-only 总是模拟）
    if len(sensor_manager.sensors) > 0:
        sensor_manager.start_simulation()


_start_demo_devices()


@app.route('/')
def demo_index():
    return render_template('demo.html')


@app.route('/demo/cameras')
def demo_cameras():
    cams = detection_engine.get_all_camera_status()
    return jsonify({'success': True, 'data': cams})


@app.route('/demo/events')
def demo_events():
    """Demo 结果流：SSE 低频推送推理结果 + 传感器快照，避免 Socket 高频更新卡顿。"""
    camera_id = request.args.get('camera_id')
    camera_id = str(camera_id) if camera_id is not None else None

    def gen():
        while True:
            try:
                payload = {
                    'ts': datetime.now().isoformat(),
                    'camera': None,
                    'sensors': None
                }

                if camera_id:
                    cam = detection_engine.cameras.get(camera_id)
                    if cam:
                        lock = detection_engine.camera_locks.get(camera_id)
                        if lock:
                            with lock:
                                last_detection = cam.get('last_detection')
                                status = cam.get('status')
                        else:
                            last_detection = cam.get('last_detection')
                            status = cam.get('status')

                        # 仅取轻量字段（避免把 thumbnail/latest_jpeg 推给前端）
                        payload['camera'] = {
                            'camera_id': camera_id,
                            'status': status,
                            'last_detection': last_detection
                        }

                # 传感器快照（轻量）
                try:
                    payload['sensors'] = sensor_manager.get_all_sensors()
                except Exception:
                    payload['sensors'] = []

                data = json.dumps(payload, ensure_ascii=False)
                yield f"data: {data}\n\n"
                time.sleep(0.05)
            except GeneratorExit:
                return
            except Exception:
                time.sleep(0.5)

    return Response(gen(), mimetype='text/event-stream')


def _mjpeg_response(camera_id: str):
    camera_id = str(camera_id)

    # 立即输出的占位帧：保证浏览器能立刻拿到首字节
    try:
        _placeholder_img = np.zeros((1, 1, 3), dtype=np.uint8)
        _ok, _buf = cv2.imencode('.jpg', _placeholder_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        placeholder_jpeg = _buf.tobytes() if _ok else b''
    except Exception:
        placeholder_jpeg = b''

    def gen():
        if placeholder_jpeg:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder_jpeg + b'\r\n')

        while True:
            try:
                jpeg = detection_engine.get_latest_jpeg(camera_id)
                if jpeg:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                else:
                    time.sleep(0.05)
            except GeneratorExit:
                return
            except Exception:
                time.sleep(0.1)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/demo/stream/<camera_id>')
def demo_stream(camera_id):
    return _mjpeg_response(camera_id)


@app.route('/stream/<camera_id>')
def stream_camera(camera_id):
    return _mjpeg_response(camera_id)

if __name__ == '__main__':
    model_status = '✓' if getattr(detection_engine, 'pipeline_available', False) else '✗'
    print(f"🚀 Demo 服务器启动 http://localhost:5000 | 摄像头: {len(detection_engine.cameras)} | 模型: {model_status}")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
