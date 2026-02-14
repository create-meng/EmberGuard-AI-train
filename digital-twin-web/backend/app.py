"""
数字孪生古建筑火灾监控Web系统 - Flask后端主应用
"""
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import os
import sys
from datetime import datetime
import warnings

# 静默警告
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# 设置静默模式（隐藏所有初始化信息）
os.environ['SILENT_MODE'] = '1'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入管理器
from config_manager import ConfigManager
from building_manager import BuildingManager
from detection_engine import DetectionEngine
from alert_manager import AlertManager
from sensor_manager import SensorManager
from history_manager import HistoryManager
from video_recorder import VideoRecorder
from api_router import APIRouter

# 尝试导入配置监听器（可选功能）
try:
    from config_watcher import ConfigWatcher
    HAS_CONFIG_WATCHER = True
except ImportError:
    HAS_CONFIG_WATCHER = False
    print("⚠️  watchdog 未安装，配置文件监听功能已禁用")

# 创建Flask应用
app = Flask(__name__, 
            static_folder='../static',
            template_folder='../frontend')

# 配置CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# 配置SocketIO
app.config['SECRET_KEY'] = 'digital-twin-fire-monitoring-secret-key'
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   logger=False, 
                   engineio_logger=False, 
                   max_decode_packets_size=10**7,
                   ping_timeout=60,
                   ping_interval=25)

# 初始化管理器
config_manager = ConfigManager()
building_manager = BuildingManager()
alert_manager = AlertManager()
sensor_manager = SensorManager(socketio=socketio, app=app)  # 关联 socketio
history_manager = HistoryManager()
video_recorder = VideoRecorder()

# 获取模型路径
config = config_manager.get_config()
models = config.get('models', {})
yolo_path = models.get('yolo', 'runs/detect/train2/weights/best.pt')
lstm_path = models.get('lstm', 'models/lstm/best.pt')

# 获取系统设置
system_settings = config.get('system', {
    'alert_enabled': True,
    'video_recording_enabled': False
})

# 初始化检测引擎
detection_engine = DetectionEngine(yolo_path, lstm_path, socketio=None, alert_manager=alert_manager, history_manager=history_manager, video_recorder=video_recorder)

# 设置告警和录制开关
detection_engine.alert_enabled = system_settings.get('alert_enabled', True)
video_recorder.recording_enabled = system_settings.get('video_recording_enabled', False)

# 初始化API路由
api_router = APIRouter(
    app, 
    config_manager, 
    building_manager, 
    detection_engine,
    alert_manager, 
    sensor_manager, 
    history_manager, 
    video_recorder
)

# 设置socketio
detection_engine.socketio = socketio
sensor_manager.socketio = socketio

# 配置文件监听器（延迟启动，在 main 中启动）
config_watcher = None

# 获取当前建筑配置
current_building_id = building_manager.get_current_building_id()
facilities = []

# 只有当有选中的建筑时才启动设备
if current_building_id:
    building_config = building_manager.load_building_config(current_building_id)
    
    # 获取设施列表（合并所有楼层，用于启动摄像头和传感器）
    if 'floors' in building_config:
        for floor in building_config.get('floors', []):
            facilities.extend(floor.get('facilities', []))
    else:
        facilities = building_config.get('facilities', [])
    
    # 从配置中注册传感器
    for facility in facilities:
        if facility.get('type') in ['temperature_sensor', 'humidity_sensor', 'smoke_detector']:
            sensor_manager.register_sensor(
                sensor_id=facility['id'],
                sensor_type=facility['type'],
                threshold=facility.get('threshold'),
                name=facility.get('name'),
                unit=facility.get('unit')
            )
    
    # 启动所有摄像头
    camera_configs = [f for f in facilities if f.get('type') == 'camera']
    use_demo = any(cam.get('demo_video') for cam in camera_configs) if camera_configs else False
    
    if camera_configs:
        detection_engine.start_all_cameras(camera_configs, use_demo)
    
    # 如果有传感器且使用演示模式，启动传感器模拟
    if len(sensor_manager.sensors) > 0 and use_demo:
        sensor_manager.start_simulation()

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    emit('connected', {'message': '连接成功'})
    
    # 仅在演示模式下自动启动传感器模拟
    # 检查是否有摄像头配置了 demo_video
    current_building_id = building_manager.get_current_building_id()
    if current_building_id:
        building_config = building_manager.load_building_config(current_building_id)
        # 获取设施列表
        facilities = []
        if 'floors' in building_config:
            for floor in building_config.get('floors', []):
                facilities.extend(floor.get('facilities', []))
        else:
            facilities = building_config.get('facilities', [])
        
        use_demo = any(cam.get('demo_video') for cam in facilities if cam.get('type') == 'camera')
        
        if use_demo and not sensor_manager.simulation_running:
            print("➤ 监测到客户端连接（演示模式），启动传感器模拟...")
            sensor_manager.start_simulation()

    # 在第一个客户端连接时启动配置监听器
    # 确保 socketio 已完全初始化
    global config_watcher
    if HAS_CONFIG_WATCHER and config_watcher is None:
        config_watcher = ConfigWatcher(socketio, building_manager, app)
        config_watcher.start()

@socketio.on('disconnect')
def handle_disconnect():
    # 客户端断开时自动离开所有房间（Socket.IO 会处理），这里保留占位
    return

@socketio.on('start_video')
def handle_start_video(data):
    camera_id = data.get('camera_id')
    if not camera_id:
        emit('video_started', {'success': False, 'error': 'missing camera_id'})
        return

    camera_id = str(camera_id)
    room = f"camera:{camera_id}"
    
    # 明确指定 sid 和 namespace
    join_room(room, sid=request.sid, namespace='/')
    print(f"SID {request.sid} 加入房间: {room}")
    
    emit('video_started', {'success': True, 'camera_id': camera_id}, namespace='/')

    # 稍微延迟一下确保 join_room 在服务器底层生效后再推帧
    def push_initial_frame():
        with app.app_context():
            try:
                status = detection_engine.get_camera_status(camera_id)
                if status:
                    has_thumb = bool(status.get('thumbnail'))
                    print(f"首帧回推确认: {camera_id} | has_thumbnail={has_thumb}")
                    socketio.emit('video_frame', {
                        'camera_id': camera_id,
                        'status': status.get('status'),
                        'thumbnail': status.get('thumbnail'),
                        'last_detection': status.get('last_detection'),
                        'timestamp': datetime.now().isoformat()
                    }, room=room, namespace='/')
            except Exception as e:
                print(f"首帧异步回推异常: {e}")

    # 使用 eventlet 或线程异步推首帧，避免阻塞当前 handle
    socketio.start_background_task(push_initial_frame)

@socketio.on('stop_video')
def handle_stop_video(data):
    camera_id = data.get('camera_id')
    if not camera_id:
        emit('video_stopped', {'success': False, 'error': 'missing camera_id'})
        return

    camera_id = str(camera_id)
    emit('video_stopped', {'success': True, 'camera_id': camera_id}, namespace='/')

if __name__ == '__main__':
    # 只在主进程输出（避免 debug 模式重复输出）
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        # 统计信息
        facilities_count = len(facilities)
        cameras_count = len([f for f in facilities if f.get('type') == 'camera'])
        buildings_count = len(building_manager.get_building_list())
        current_building = building_manager.get_current_building_id() or '无'
        model_status = '✓' if getattr(detection_engine, 'pipeline_available', False) else '✗'
        
        print(f"🚀 服务器启动 http://localhost:5000 | 当前建筑: {current_building} | 设施: {facilities_count} | 摄像头: {cameras_count} | 建筑总数: {buildings_count} | 模型: {model_status}")
    
    # 监控配置文件变化，自动重启
    extra_files = [
        '../config/system_config.json',
        '../config/hardware_config.json',
        '../config/buildings.json'
    ]
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True, 
        use_reloader=False,
        allow_unsafe_werkzeug=True,
        extra_files=extra_files,
        log_output=False
    )
