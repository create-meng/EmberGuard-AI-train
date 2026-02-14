"""
AI检测引擎 - 集成YOLO+LSTM火灾检测模型
"""
import threading
import time
import cv2
import base64
from typing import Optional
import numpy as np
from datetime import datetime
from collections import deque
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from emberguard.pipeline import FireDetectionPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入FireDetectionPipeline: {e}")
    print("检测引擎将以演示模式运行")
    PIPELINE_AVAILABLE = False


class DetectionEngine:
    """AI检测引擎 - 集成YOLO+LSTM火灾检测"""
    
    def __init__(self, yolo_path, lstm_path, socketio=None, alert_manager=None, history_manager=None, video_recorder=None):
        """
        初始化检测引擎
        
        Args:
            yolo_path: YOLO模型路径
            lstm_path: LSTM模型路径
            socketio: SocketIO实例（用于推送数据）
            alert_manager: 告警管理器
            history_manager: 历史数据管理器
            video_recorder: 视频录制器
        """
        self.yolo_path = yolo_path
        self.lstm_path = lstm_path
        self.socketio = socketio
        self.alert_manager = alert_manager
        self.history_manager = history_manager
        self.video_recorder = video_recorder

        # 视频订阅推送控制
        self.video_push_interval = 0.25

        # 推理节流：即使推理很慢，也保证视频缩略图持续输出
        self.infer_interval = 0.5
        
        # 系统开关
        self.alert_enabled = True  # 告警开关
        
        # 摄像头管理
        self.cameras = {}  # camera_id -> camera_info
        self.running_cameras = {}  # camera_id -> thread
        self.camera_locks = {}  # camera_id -> lock
        
        # 智能帧率控制
        self.normal_fps = 5
        self.alert_fps = 15
        self.camera_fps = {}  # camera_id -> current_fps
        self.alert_cooldown = 5  # 告警后5秒恢复正常帧率
        self.last_alert_time = {}  # camera_id -> timestamp
        
        # 告警冷却（避免同一摄像头短时间内重复告警）
        self.alert_interval = 10  # 同一摄像头10秒内只能告警一次
        self.last_alert_sent = {}  # camera_id -> timestamp
        
        # 检测管道
        # 每个摄像头独立 pipeline，避免 LSTM 缓冲串台
        self._model_paths = {
            'yolo': yolo_path,
            'lstm': lstm_path
        }
        self.camera_pipelines = {}  # camera_id -> FireDetectionPipeline
        self.pipeline_available = bool(PIPELINE_AVAILABLE)
        if not self.pipeline_available and not os.environ.get('SILENT_MODE'):
            print("警告: 检测管道不可用，无法进行 YOLO+LSTM 推理")
    
    def start_all_cameras(self, camera_configs, use_demo_video=False):
        """
        启动所有摄像头
        
        Args:
            camera_configs: 摄像头配置列表
            use_demo_video: 是否使用演示视频（自动检测到 demo_video 字段时为 True）
        """
        print(f"\n启动所有摄像头 (使用演示视频: {use_demo_video})")
        
        for cam_config in camera_configs:
            camera_id = str(cam_config['id'])
            
            # 选择视频源
            if use_demo_video and 'demo_video' in cam_config:
                source = cam_config['demo_video']
            else:
                source = cam_config.get('source', 0)
            
            # 启动摄像头
            self.start_camera(camera_id, source, cam_config)
        
        print(f"✓ 已启动 {len(camera_configs)} 个摄像头\n")
    
    def start_camera(self, camera_id, source, config=None):
        """
        启动单个摄像头检测
        
        Args:
            camera_id: 摄像头ID
            source: 视频源（设备索引、RTSP地址或视频文件）
            config: 摄像头配置
        """
        camera_id = str(camera_id)

        # 如果已经在运行，先停止
        if camera_id in self.running_cameras:
            self.stop_camera(camera_id)
        
        # 初始化摄像头信息
        self.cameras[camera_id] = {
            'id': camera_id,
            'name': config.get('name', camera_id) if config else camera_id,
            'source': source,
            'status': 'starting',
            'fps': 0,
            'last_detection': None,
            'thumbnail': None,
            'latest_jpeg': None,
            'latest_jpeg_ts': None,
            'config': config or {}
        }

        # 为每个摄像头创建独立 pipeline（确保 LSTM buffer 隔离）
        # 要求：无论 demo/非 demo 都必须走真实推理；demo 仅改变输入源
        if not self.pipeline_available:
            self.cameras[camera_id]['status'] = 'error'
        else:
            try:
                self.camera_pipelines[camera_id] = FireDetectionPipeline(
                    yolo_model_path=self._model_paths['yolo'],
                    lstm_model_path=self._model_paths['lstm'],
                    sequence_length=30
                )
            except Exception as e:
                print(f"✗ 摄像头 {camera_id} 模型加载失败: {e}")
                self.cameras[camera_id]['status'] = 'error'
        
        # 创建锁
        self.camera_locks[camera_id] = threading.Lock()
        
        # 初始化帧率
        self.camera_fps[camera_id] = self.normal_fps
        self.last_alert_time[camera_id] = 0
        
        # 启动检测线程
        thread = threading.Thread(
            target=self._camera_loop,
            args=(camera_id,),
            daemon=True
        )
        thread.start()
        self.running_cameras[camera_id] = thread
        
        print(f"✓ 摄像头 {camera_id} 启动中...")
    
    def stop_camera(self, camera_id):
        """
        停止摄像头检测
        
        Args:
            camera_id: 摄像头ID
        """
        camera_id = str(camera_id)

        if camera_id in self.cameras:
            self.cameras[camera_id]['status'] = 'stopped'
        
        if camera_id in self.running_cameras:
            # 等待线程结束（最多2秒）
            thread = self.running_cameras[camera_id]
            thread.join(timeout=2.0)
            del self.running_cameras[camera_id]
        
        print(f"✓ 摄像头 {camera_id} 已停止")

        if camera_id in self.camera_pipelines:
            del self.camera_pipelines[camera_id]
    
    def _camera_loop(self, camera_id):
        """
        摄像头检测循环（独立线程）
        
        Args:
            camera_id: 摄像头ID
        """
        camera_id = str(camera_id)
        camera_info = self.cameras[camera_id]
        source = camera_info['source']

        pipeline = self.camera_pipelines.get(camera_id)
        if camera_info.get('status') == 'error' or pipeline is None:
            print(f"✗ 摄像头 {camera_id} 无可用模型，无法推理")
            camera_info['status'] = 'error'
            self._push_camera_update(camera_id)
            return
        
        # 打开视频源
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"✗ 无法打开摄像头 {camera_id}: {source}")
            camera_info['status'] = 'offline'
            return
        
        print(f"✓ 摄像头 {camera_id} 已连接")
        camera_info['status'] = 'online'
        
        # 重置LSTM缓冲区
        try:
            pipeline.reset_buffer()
        except Exception:
            pass
        
        frame_count = 0
        last_push_time = time.time()
        last_video_push_time = 0
        last_infer_time = 0
        
        try:
            while camera_info['status'] != 'stopped':
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    # 视频文件结束，循环播放
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    try:
                        pipeline.reset_buffer()
                    except Exception:
                        pass
                    continue

                if frame is None:
                    if not camera_info.get('_logged_frame_none'):
                        print(f"✗ 摄像头 {camera_id} 读取到空帧(frame=None)")
                        camera_info['_logged_frame_none'] = True
                    continue
                
                # 统一调整帧尺寸为640x480，避免LSTM输入尺寸不匹配
                try:
                    frame_resized = cv2.resize(frame, (640, 480))
                except Exception as e:
                    if not camera_info.get('_logged_resize_error'):
                        print(f"✗ 摄像头 {camera_id} resize失败: {e}")
                        camera_info['_logged_resize_error'] = True
                    continue

                # 先生成缩略图，确保前端画面不会被推理阻塞
                thumbnail = self._generate_thumbnail(frame_resized, 640, 480)

                # 同步缓存 MJPEG 使用的 JPEG 二进制（不做 base64）
                latest_jpeg = self._encode_jpeg_bytes(frame_resized)

                # 诊断：确认是否能生成首帧缩略图
                if thumbnail and not camera_info.get('_logged_first_thumbnail'):
                    print(f"✓ 摄像头 {camera_id} 已生成首帧thumbnail")
                    camera_info['_logged_first_thumbnail'] = True
                if (not thumbnail) and (not camera_info.get('_logged_thumbnail_none')):
                    print(f"✗ 摄像头 {camera_id} thumbnail生成失败(None)")
                    camera_info['_logged_thumbnail_none'] = True
                
                # 获取当前帧率
                current_fps = self.camera_fps.get(camera_id, self.normal_fps)
                
                # 帧率控制
                frame_count += 1
                should_process = (frame_count % max(1, int(30 / current_fps)) == 0)

                # 推理节流：按时间间隔执行真实 YOLO+LSTM 推理
                detection_result = None
                current_time = time.time()
                if should_process and (current_time - last_infer_time >= self.infer_interval):
                    detection_result = self._detect_frame(frame_resized, camera_id)
                    last_infer_time = current_time
                
                # 更新摄像头信息
                with self.camera_locks[camera_id]:
                    if detection_result is not None:
                        camera_info['last_detection'] = detection_result
                    camera_info['fps'] = current_fps
                    camera_info['thumbnail'] = thumbnail
                    if latest_jpeg is not None:
                        camera_info['latest_jpeg'] = latest_jpeg
                        camera_info['latest_jpeg_ts'] = datetime.now().isoformat()
                
                # 检查是否需要触发告警
                if detection_result is not None:
                    self._check_alert(camera_id, detection_result)
                
                # 推送轻量状态（每0.5秒推送一次）
                if current_time - last_push_time >= 0.5:
                    self._push_camera_update(camera_id)
                    last_push_time = current_time

                # Socket.IO 视频帧推送已弃用：视频走 HTTP MJPEG 流。
                # 这里不再做高频 emit，避免大包导致连接不稳定。
                
                # 智能帧率调整
                if detection_result is not None:
                    self._adjust_fps(camera_id, detection_result)
                
                # 保存历史数据
                if self.history_manager:
                    if detection_result is not None:
                        self.history_manager.save_detection_record(camera_id, detection_result)
                
                # 保存视频帧到循环缓冲区（使用调整后的帧）
                if self.video_recorder:
                    self.video_recorder.save_frame(camera_id, frame_resized, detection_result)
                
        except Exception as e:
            print(f"✗ 摄像头 {camera_id} 检测循环异常: {e}")
            camera_info['status'] = 'error'
        finally:
            cap.release()
            if camera_info.get('status') != 'error':
                camera_info['status'] = 'offline'
            print(f"✓ 摄像头 {camera_id} 检测循环结束")
    
    def _detect_frame(self, frame, camera_id):
        """
        检测单帧
        
        Args:
            frame: 视频帧
            camera_id: 摄像头ID
            
        Returns:
            dict: 检测结果
        """
        pipeline = self.camera_pipelines.get(camera_id)
        if pipeline is None:
            # 无模型不产生演示随机数据，直接返回空
            return None

        try:
            # 使用YOLO+LSTM检测
            result = pipeline.detect_frame(frame, conf_threshold=0.25)

            # 格式化结果
            return {
                'timestamp': datetime.now().isoformat(),
                'yolo_detections': result.get('yolo_detections', []),
                'has_detection': result.get('has_detection', False),
                'lstm_prediction': result.get('lstm_prediction'),
                'lstm_class_name': result.get('lstm_class_name'),
                'lstm_confidence': result.get('lstm_confidence'),
                'lstm_probabilities': result.get('lstm_probabilities', {}),
                'buffer_size': result.get('buffer_size', 0)
            }
        except Exception as e:
            print(f"检测异常: {e}")
            return None

    def _encode_jpeg_bytes(self, frame, quality: int = 85) -> Optional[bytes]:
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            ok, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not ok:
                return None
            return buffer.tobytes()
        except Exception:
            return None

    def get_latest_jpeg(self, camera_id) -> Optional[bytes]:
        camera_id = str(camera_id)
        if camera_id not in self.cameras:
            return None
        if camera_id not in self.camera_locks:
            return None
        with self.camera_locks[camera_id]:
            return self.cameras[camera_id].get('latest_jpeg')
    
    def _generate_thumbnail(self, frame, width, height):
        """
        生成缩略图
        
        Args:
            frame: 原始帧
            width: 缩略图宽度
            height: 缩略图高度
            
        Returns:
            str: base64编码的JPEG图像
        """
        try:
            # 如果帧尺寸已经是目标尺寸，直接使用
            if frame.shape[1] == width and frame.shape[0] == height:
                thumbnail = frame
            else:
                # 调整大小
                thumbnail = cv2.resize(frame, (width, height))
            
            # 编码为JPEG
            _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # 转换为base64
            thumbnail_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return f"data:image/jpeg;base64,{thumbnail_b64}"
        except Exception as e:
            print(f"生成缩略图失败: {e}")
            return None
    
    def _check_alert(self, camera_id, detection_result):
        """
        检查是否需要触发告警
        
        Args:
            camera_id: 摄像头ID
            detection_result: 检测结果
        """
        if not self.alert_manager:
            return

        if not detection_result or not isinstance(detection_result, dict):
            return
        
        lstm_pred = detection_result.get('lstm_prediction')
        lstm_conf = detection_result.get('lstm_confidence', 0)
        
        # 烟雾或火焰且置信度>0.7
        if lstm_pred in [1, 2] and lstm_conf > 0.7:
            # 检查告警冷却时间
            current_time = time.time()
            last_sent = self.last_alert_sent.get(camera_id, 0)
            
            if current_time - last_sent < self.alert_interval:
                # 冷却期内，不发送告警
                return
            
            camera_info = self.cameras.get(camera_id, {})
            
            alert_data = {
                'camera_id': camera_id,
                'camera_name': camera_info.get('name', camera_id),
                'type': 'smoke' if lstm_pred == 1 else 'fire',
                'confidence': lstm_conf,
                'timestamp': detection_result.get('timestamp'),
                'lstm_class': detection_result.get('lstm_class_name')
            }
            
            # 检查告警开关：只有开启时才添加告警记录和推送通知
            if self.alert_enabled:
                # 添加告警记录
                self.alert_manager.add_alert(alert_data)
                
                # 通过WebSocket推送告警通知
                if self.socketio:
                    self.socketio.emit('new_alert', alert_data)
            
            # 标记告警时间点，保存视频快照（受视频录制开关控制）
            if self.video_recorder:
                self.video_recorder.mark_alert(camera_id, datetime.now())
            
            # 更新最后告警时间
            self.last_alert_sent[camera_id] = current_time
    
    def _adjust_fps(self, camera_id, detection_result):
        """
        智能帧率调整
        
        Args:
            camera_id: 摄像头ID
            detection_result: 检测结果
        """
        if not detection_result or not isinstance(detection_result, dict):
            return

        lstm_pred = detection_result.get('lstm_prediction')
        current_time = time.time()
        
        # 检测到异常：提高帧率
        if lstm_pred in [1, 2]:
            self.camera_fps[camera_id] = self.alert_fps
            self.last_alert_time[camera_id] = current_time
        else:
            # 检查是否需要恢复正常帧率
            last_alert = self.last_alert_time.get(camera_id, 0)
            if current_time - last_alert > self.alert_cooldown:
                self.camera_fps[camera_id] = self.normal_fps
    
    def _push_camera_update(self, camera_id):
        """
        推送摄像头状态更新
        
        Args:
            camera_id: 摄像头ID
        """
        if not self.socketio:
            return
        
        camera_id = str(camera_id)
        camera_info = self.cameras.get(camera_id)
        if not camera_info:
            return
        
        with self.camera_locks[camera_id]:
            update_data = {
                'camera_id': str(camera_id),
                'camera_name': camera_info.get('name'),
                'status': camera_info.get('status'),
                'fps': camera_info.get('fps'),
                'last_detection': camera_info.get('last_detection'),
                'timestamp': datetime.now().isoformat()
            }
        
        self.socketio.emit('camera_update', update_data, namespace='/')

    def _push_video_frame(self, camera_id):
        # 已弃用：视频帧不再通过 Socket.IO 发送。
        return
    
    def get_all_camera_status(self):
        """获取所有摄像头状态"""
        status_list = []
        
        for camera_id, camera_info in self.cameras.items():
            with self.camera_locks.get(camera_id, threading.Lock()):
                status_list.append({
                    'id': camera_id,
                    'name': camera_info.get('name'),
                    'status': camera_info.get('status'),
                    'fps': camera_info.get('fps'),
                    'last_detection': camera_info.get('last_detection')
                })
        
        return status_list
    
    def get_camera_info(self, camera_id):
        """获取摄像头信息"""
        camera_id = str(camera_id)
        camera_info = self.cameras.get(camera_id)
        if not camera_info:
            return None
        
        with self.camera_locks.get(camera_id, threading.Lock()):
            return {
                'id': camera_id,
                'name': camera_info.get('name'),
                'status': camera_info.get('status'),
                'source': camera_info.get('source')
            }
    
    def get_camera_status(self, camera_id):
        """获取摄像头实时状态"""
        camera_id = str(camera_id)
        camera_info = self.cameras.get(camera_id)
        if not camera_info:
            return None
        
        with self.camera_locks.get(camera_id, threading.Lock()):
            return {
                'id': camera_id,
                'name': camera_info.get('name'),
                'status': camera_info.get('status'),
                'fps': camera_info.get('fps'),
                'last_detection': camera_info.get('last_detection'),
                'thumbnail': camera_info.get('thumbnail')
            }
