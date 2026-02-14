"""数字孪生演示系统 - Flask 后端"""
from flask import Flask, render_template, Response
import os
import sys
import json
from datetime import datetime
import warnings
import time
import cv2
import numpy as np

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
yolo_path = models.get('yolo', 'runs/detect/train2/weights/best.pt')
lstm_path = models.get('lstm', 'models/lstm/best.pt')

# 初始化检测引擎
detection_engine = DetectionEngine(
    yolo_path,
    lstm_path,
)


def _start_demo_devices():
    # 单路演示：固定视频源 + 单个传感器
    demo_video = r"D:\a安建大\大二\下学期\比赛\挑战杯\院赛\AI消防\ultralytics-main\datasets\fire_videos_organized\fire\archive_fire2.mp4"

    if not os.path.exists(demo_video):
        print(f"✗ Demo 视频不存在: {demo_video}")
    else:
        detection_engine.start(demo_video, name='主摄像头')

    # 单个传感器（演示模式：始终模拟）
    sensor_manager.register_sensor(
        sensor_id='sensor_temp_001',
        sensor_type='temperature_sensor',
        threshold=60,
        name='温度传感器',
        unit='°C'
    )
    sensor_manager.start_simulation()


_start_demo_devices()


@app.route('/')
def demo_index():
    return render_template('demo.html')


@app.route('/demo/events')
def demo_events():
    """Demo 结果流：SSE 低频推送推理结果 + 传感器快照，避免 Socket 高频更新卡顿。"""
    def gen():
        while True:
            try:
                payload = {
                    'ts': datetime.now().strftime('%H:%M:%S'),
                    'camera': None,
                    'sensors': None
                }

                payload['camera'] = detection_engine.get_snapshot()

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


def _mjpeg_response():
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
                jpeg = detection_engine.get_latest_jpeg()
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


@app.route('/demo/stream')
def demo_stream():
    return _mjpeg_response()

if __name__ == '__main__':
    model_status = '✓' if getattr(detection_engine, 'pipeline_available', False) else '✗'
    print(f"🚀 Demo 服务器启动 http://localhost:5000 | 摄像头: demo_cam_001 | 模型: {model_status}")
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
