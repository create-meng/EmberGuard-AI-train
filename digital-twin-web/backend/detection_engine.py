"""
AI检测引擎 - 集成YOLO+LSTM火灾检测模型
"""
import threading
import time
import cv2
from typing import Optional
from datetime import datetime
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

        # 推理节流：推理在后台线程执行，避免阻塞视频帧输出
        self.infer_interval = 0.033

        # MJPEG 输出帧率（只影响 latest_jpeg 更新频率）
        self.stream_fps = 30

        # CPU 加速：推理用更小尺寸，bbox 缩放回 640x480
        self.infer_size = (320, 240)

        # 终端输出节流
        self.log_interval_sec = 1.0

        self.pipeline_available = bool(PIPELINE_AVAILABLE)
        if not self.pipeline_available and not os.environ.get('SILENT_MODE'):
            print("警告: 检测管道不可用，无法进行 YOLO+LSTM 推理")

        # 单摄像头状态
        self._lock = threading.Lock()
        self._pipeline = None
        self._thread = None
        self._infer_busy = False
        self._source = None
        self._name = '主摄像头'
        self._status = 'idle'
        self._latest_jpeg: Optional[bytes] = None
        self._latest_jpeg_ts: Optional[str] = None
        self._last_detection = None

    def start(self, source: str, name: str = '主摄像头'):
        """启动单路视频源推理线程（demo-only）。"""
        self._source = source
        self._name = name

        if not self.pipeline_available:
            with self._lock:
                self._status = 'error'
            return

        try:
            self._pipeline = FireDetectionPipeline(
                yolo_model_path=self.yolo_path,
                lstm_model_path=self.lstm_path,
                sequence_length=30,
            )
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            with self._lock:
                self._status = 'error'
            return

        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()
    
    def _camera_loop(self):
        """
        摄像头检测循环（独立线程）
        """
        pipeline = self._pipeline
        source = self._source
        if pipeline is None or not source:
            with self._lock:
                self._status = 'error'
            return
        
        # 打开视频源
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"✗ 无法打开视频源: {source}")
            with self._lock:
                self._status = 'offline'
            return

        print("✓ 视频源已连接")
        with self._lock:
            self._status = 'online'
        
        # 重置LSTM缓冲区
        try:
            pipeline.reset_buffer()
        except Exception:
            pass

        last_infer_time = 0
        last_frame_time = 0
        last_log_time = 0
        last_infer_ms = None
        
        logged_resize_error = False

        try:
            while True:
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
                    continue
                
                # 统一调整帧尺寸为640x480，避免LSTM输入尺寸不匹配
                try:
                    frame_resized = cv2.resize(frame, (640, 480))
                except Exception as e:
                    if not logged_resize_error:
                        print(f"✗ resize失败: {e}")
                        logged_resize_error = True
                    continue

                # MJPEG：只在需要刷新画面时才编码 JPEG，避免每帧都 imencode 抢 CPU
                
                current_time = time.time()

                # 推理节流：后台线程执行，避免阻塞 MJPEG
                # - 只要到达 infer_interval 且当前没有推理任务，就启动一次
                if current_time - last_infer_time >= self.infer_interval:
                    with self._lock:
                        infer_busy = bool(self._infer_busy)
                    if not infer_busy:
                        # 推理用更小的分辨率，加速 CPU 推理
                        try:
                            frame_for_infer = cv2.resize(frame_resized, self.infer_size)
                            sx = 640.0 / float(self.infer_size[0])
                            sy = 480.0 / float(self.infer_size[1])
                        except Exception:
                            frame_for_infer = frame_resized.copy()
                            sx, sy = 1.0, 1.0

                        def _infer_job():
                            try:
                                _t0 = time.time()
                                res = self._detect_frame(frame_for_infer)
                                if res is None:
                                    return
                                try:
                                    res['infer_ms'] = int((time.time() - _t0) * 1000)
                                except Exception:
                                    pass

                                # bbox 坐标缩放回 640x480（前端固定按 640x480 解释）
                                try:
                                    yolo = res.get('yolo_detections')
                                    if isinstance(yolo, list) and (sx != 1.0 or sy != 1.0):
                                        for det in yolo:
                                            bb = det.get('bbox') if isinstance(det, dict) else None
                                            if not bb or len(bb) != 4:
                                                continue
                                            x1, y1, x2, y2 = bb
                                            det['bbox'] = [
                                                int(x1 * sx),
                                                int(y1 * sy),
                                                int(x2 * sx),
                                                int(y2 * sy),
                                            ]
                                except Exception:
                                    pass

                                with self._lock:
                                    self._last_detection = res
                            finally:
                                with self._lock:
                                    self._infer_busy = False

                        with self._lock:
                            self._infer_busy = True
                        t = threading.Thread(target=_infer_job, daemon=True)
                        t.start()
                        last_infer_time = current_time

                # MJPEG：按 stream_fps 更新 latest_jpeg，保证画面流畅
                stream_interval = 1.0 / max(1, int(self.stream_fps))
                if (current_time - last_frame_time) >= stream_interval:
                    latest_jpeg = self._encode_jpeg_bytes(frame_resized, quality=80)
                    with self._lock:
                        if latest_jpeg is not None:
                            self._latest_jpeg = latest_jpeg
                            self._latest_jpeg_ts = datetime.now().strftime('%H:%M:%S')
                    last_frame_time = current_time

                with self._lock:
                    _det = self._last_detection

                # 终端输出：让你看到推理是否在跑、耗时多少、LSTM 缓冲进度
                if (current_time - last_log_time) >= self.log_interval_sec:
                    try:
                        if isinstance(_det, dict):
                            last_infer_ms = _det.get('infer_ms', last_infer_ms)
                            buf = _det.get('buffer_size', '-')
                            ycnt = len(_det.get('yolo_detections', []) or [])
                        else:
                            buf = '-'
                            ycnt = 0
                        ms_txt = f"{last_infer_ms}ms" if last_infer_ms is not None else "-"
                        print(f"[DEMO] infer={ms_txt} | yolo={ycnt} | lstm_buf={buf}/30")
                    except Exception:
                        pass
                    last_log_time = current_time
                
                # 轻微 sleep 防止 CPU 空转，同时不影响流畅度
                time.sleep(0.001)
                
        except Exception as e:
            print(f"✗ 检测循环异常: {e}")
            with self._lock:
                self._status = 'error'
        finally:
            cap.release()
            with self._lock:
                if self._status != 'error':
                    self._status = 'offline'
            print("✓ 检测循环结束")
    
    def _detect_frame(self, frame):
        """
        检测单帧
        
        Args:
            frame: 视频帧
            
        Returns:
            dict: 检测结果
        """
        pipeline = self._pipeline
        if pipeline is None:
            # 无模型不产生演示随机数据，直接返回空
            return None

        try:
            # 使用YOLO+LSTM检测
            result = pipeline.detect_frame(frame, conf_threshold=0.25)

            # 格式化结果
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
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

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                'camera_id': 'demo_cam_001',
                'camera_name': self._name,
                'status': self._status,
                'last_detection': self._last_detection,
            }
    

