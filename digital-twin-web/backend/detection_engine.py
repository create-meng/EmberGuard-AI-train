"""
AI检测引擎 - 集成YOLO+LSTM火灾检测模型
"""
import threading
import time
import cv2
import numpy as np
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
    
    def __init__(
        self,
        yolo_path,
        lstm_path,
        socketio=None,
        alert_manager=None,
        history_manager=None,
        video_recorder=None,
        use_lstm: bool = True,
        enable_feature_denoise: bool = True,
        enable_frame_denoise: bool = True,
        enable_fusion: bool = False,
        experiment_profile: str = 'yolo_lstm_denoise_fusion',
    ):
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

        self.use_lstm = bool(use_lstm)
        self.enable_feature_denoise = bool(enable_feature_denoise)

        self.enable_fusion = bool(enable_fusion)
        self.experiment_profile = str(experiment_profile) if experiment_profile is not None else 'yolo_lstm_denoise_fusion'

        # 推理节流：空闲/有人观看时不同频率。
        # 边缘设备更怕“打开网页后负载骤增”，因此 active 默认更保守。
        self.infer_interval_idle = 0.25
        self.infer_interval_active = 0.22
        self.infer_interval = self.infer_interval_idle

        # MJPEG 输出：没有观看者时不做 JPEG 编码（或极低频），大幅降低 CPU 开销
        self.stream_fps_idle = 0
        self.stream_fps_active = 6
        self.stream_fps = self.stream_fps_idle

        self.stream_jpeg_quality_active = 72

        # CPU 加速：推理用更小尺寸，bbox 缩放回 640x480
        self.infer_size = (320, 240)

        self.enable_frame_denoise = bool(enable_frame_denoise)
        self.frame_denoise_alpha = 0.78
        self._frame_denoise_state = None

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
        self._camera_id = 'demo_cam_001'  # 默认摄像头ID
        self._status = 'idle'
        self._latest_jpeg: Optional[bytes] = None
        self._latest_jpeg_ts: Optional[str] = None
        self._last_detection = None
        self._stream_clients = 0

        # fusion 状态（每个引擎/摄像头独立）
        self._fusion_hist = []
        self._fusion_alarm_state = 'normal'
        self._fusion_alarm_since = 0

        # fusion 参数（与前端 demo.js 对齐，保持最小可用）
        self._fusion_window_size = 20
        self._fusion_on_fire_ratio = 0.35
        self._fusion_off_fire_ratio = 0.18
        self._fusion_on_smoke_ratio = 0.45
        self._fusion_off_smoke_ratio = 0.22
        self._fusion_yolo_strong_conf = 0.7
        self._fusion_yolo_strong_min_area = 0.008

    def _yolo_evidence(self, dets, frame_w: int = 640, frame_h: int = 480):
        fire = 0
        smoke = 0
        fire_strong = 0
        smoke_strong = 0
        if not isinstance(dets, list):
            return {'fire': 0, 'smoke': 0, 'fire_strong': 0, 'smoke_strong': 0}
        for d in dets:
            if not isinstance(d, dict):
                continue
            cls = d.get('class_name')
            conf = d.get('confidence')
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0

            area = 0.0
            bb = d.get('bbox')
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                try:
                    x1, y1, x2, y2 = [float(x) for x in bb]
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    denom = float(max(1, frame_w * frame_h))
                    area = (w * h) / denom
                except Exception:
                    area = 0.0

            if cls == 'fire':
                if conf >= 0.25:
                    fire += 1
                if conf >= self._fusion_yolo_strong_conf and area >= self._fusion_yolo_strong_min_area:
                    fire_strong += 1
            elif cls == 'smoke':
                if conf >= 0.25:
                    smoke += 1
                if conf >= self._fusion_yolo_strong_conf and area >= 0.01:
                    smoke_strong += 1
        return {'fire': fire, 'smoke': smoke, 'fire_strong': fire_strong, 'smoke_strong': smoke_strong}

    def _decide_without_fusion(self, yolo_dets, lstm_pred):
        if not self.use_lstm:
            has_fire = any(isinstance(x, dict) and x.get('class_name') == 'fire' for x in (yolo_dets or []))
            has_smoke = any(isinstance(x, dict) and x.get('class_name') == 'smoke' for x in (yolo_dets or []))
            level = 'fire' if has_fire else ('smoke' if has_smoke else 'normal')
            reason = 'YOLO: 有框' if level != 'normal' else 'YOLO: 无框'
            return level, reason, 'yolo'

        has_lstm = lstm_pred is not None
        if not has_lstm:
            has_fire = any(isinstance(x, dict) and x.get('class_name') == 'fire' for x in (yolo_dets or []))
            has_smoke = any(isinstance(x, dict) and x.get('class_name') == 'smoke' for x in (yolo_dets or []))
            level = 'fire' if has_fire else ('smoke' if has_smoke else 'normal')
            return level, 'LSTM未就绪: YOLO兜底', 'yolo_fallback'

        try:
            pred = int(lstm_pred)
        except Exception:
            pred = -1
        if pred == 2:
            return 'fire', 'LSTM: 直接输出', 'lstm'
        if pred == 1:
            return 'smoke', 'LSTM: 直接输出', 'lstm'
        return 'normal', 'LSTM: 直接输出', 'lstm'

    def _decide_with_fusion(self, yolo_dets, lstm_pred, lstm_confidence):
        now = int(time.time() * 1000)
        y = self._yolo_evidence(yolo_dets)

        if y.get('fire_strong', 0) > 0:
            self._fusion_alarm_state = 'fire'
            self._fusion_alarm_since = now
            return 'fire', f"YOLO 强证据({int(y.get('fire_strong', 0))})", 'yolo_strong'

        has_lstm = lstm_pred is not None
        src = 'yolo'
        vote_fire = 0
        vote_smoke = 0
        if has_lstm:
            src = 'lstm'
            try:
                pred = int(lstm_pred)
            except Exception:
                pred = -1
            vote_fire = 1 if pred == 2 else 0
            vote_smoke = 1 if pred == 1 else 0
        else:
            vote_fire = 1 if y.get('fire', 0) > 0 else 0
            vote_smoke = 1 if y.get('smoke', 0) > 0 else 0

        self._fusion_hist.append({
            'fire': vote_fire,
            'smoke': vote_smoke,
            'src': src,
            't': now,
            'conf': float(lstm_confidence) if isinstance(lstm_confidence, (int, float)) else None,
        })
        if len(self._fusion_hist) > int(self._fusion_window_size):
            self._fusion_hist = self._fusion_hist[-int(self._fusion_window_size):]

        n = len(self._fusion_hist)
        fire_votes = sum(1 for x in self._fusion_hist if x.get('fire'))
        smoke_votes = sum(1 for x in self._fusion_hist if x.get('smoke'))
        fire_ratio = (fire_votes / n) if n else 0.0
        smoke_ratio = (smoke_votes / n) if n else 0.0

        next_state = self._fusion_alarm_state or 'normal'
        reason = ''
        if next_state == 'fire':
            if fire_ratio <= self._fusion_off_fire_ratio:
                next_state = 'normal'
                reason = f"解除fire: ratio={fire_ratio:.2f} src={src.upper()}"
            else:
                reason = f"保持fire: ratio={fire_ratio:.2f} src={src.upper()}"
        elif next_state == 'smoke':
            if fire_ratio >= self._fusion_on_fire_ratio:
                next_state = 'fire'
                reason = f"升级fire: ratio={fire_ratio:.2f} src={src.upper()}"
            elif smoke_ratio <= self._fusion_off_smoke_ratio:
                next_state = 'normal'
                reason = f"解除smoke: ratio={smoke_ratio:.2f} src={src.upper()}"
            else:
                reason = f"保持smoke: ratio={smoke_ratio:.2f} src={src.upper()}"
        else:
            if fire_ratio >= self._fusion_on_fire_ratio:
                next_state = 'fire'
                reason = f"触发fire: ratio={fire_ratio:.2f} src={src.upper()}"
            elif smoke_ratio >= self._fusion_on_smoke_ratio:
                next_state = 'smoke'
                reason = f"触发smoke: ratio={smoke_ratio:.2f} src={src.upper()}"
            else:
                reason = f"normal: fire={fire_ratio:.2f} smoke={smoke_ratio:.2f} src={src.upper()}"

        self._fusion_alarm_state = next_state
        if next_state == 'normal':
            self._fusion_alarm_since = 0
        else:
            self._fusion_alarm_since = self._fusion_alarm_since or now

        return next_state, reason, 'fusion'

    def _denoise_frame_for_infer(self, frame_small):
        if not self.enable_frame_denoise:
            return frame_small
        if frame_small is None:
            return frame_small
        try:
            a = float(self.frame_denoise_alpha)
            if a < 0.0:
                a = 0.0
            elif a > 1.0:
                a = 1.0

            if self._frame_denoise_state is None:
                self._frame_denoise_state = frame_small.astype(np.float32)
            else:
                if self._frame_denoise_state.shape[:2] != frame_small.shape[:2]:
                    self._frame_denoise_state = frame_small.astype(np.float32)
                cv2.accumulateWeighted(frame_small, self._frame_denoise_state, a)
            out = np.clip(self._frame_denoise_state, 0, 255).astype(np.uint8)
            return out
        except Exception:
            return frame_small

    def add_stream_client(self):
        with self._lock:
            self._stream_clients = int(self._stream_clients) + 1

    def remove_stream_client(self):
        with self._lock:
            self._stream_clients = max(0, int(self._stream_clients) - 1)

    def _is_streaming_active(self) -> bool:
        with self._lock:
            return int(self._stream_clients) > 0

    def start(self, source: str, name: str = '主摄像头', camera_id: str = 'demo_cam_001'):
        """启动单路视频源推理线程（demo-only）。"""
        self._source = source
        self._name = name
        self._camera_id = camera_id

        if not self.pipeline_available:
            with self._lock:
                self._status = 'error'
            return

        try:
            self._pipeline = FireDetectionPipeline(
                yolo_model_path=self.yolo_path,
                lstm_model_path=(self.lstm_path if self.use_lstm else None),
                sequence_length=30,
                enable_feature_denoise=self.enable_feature_denoise,
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

                # 根据是否有人观看动态调整推理/编码频率
                streaming_active = self._is_streaming_active()
                if streaming_active:
                    self.infer_interval = self.infer_interval_active
                    self.stream_fps = self.stream_fps_active
                else:
                    self.infer_interval = self.infer_interval_idle
                    self.stream_fps = self.stream_fps_idle

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

                        frame_for_infer = self._denoise_frame_for_infer(frame_for_infer)

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
                if int(self.stream_fps) > 0:
                    stream_interval = 1.0 / max(1, int(self.stream_fps))
                    if (current_time - last_frame_time) >= stream_interval:
                        q = self.stream_jpeg_quality_active
                        latest_jpeg = self._encode_jpeg_bytes(frame_resized, quality=q)
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

            yolo_dets = result.get('yolo_detections', [])
            lstm_pred = result.get('lstm_prediction')
            lstm_conf = result.get('lstm_confidence')

            with self._lock:
                if self.enable_fusion:
                    final_alarm, final_reason, final_source = self._decide_with_fusion(yolo_dets, lstm_pred, lstm_conf)
                else:
                    final_alarm, final_reason, final_source = self._decide_without_fusion(yolo_dets, lstm_pred)

            # 格式化结果
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'yolo_detections': yolo_dets,
                'has_detection': result.get('has_detection', False),
                'lstm_prediction': result.get('lstm_prediction'),
                'lstm_class_name': result.get('lstm_class_name'),
                'lstm_confidence': result.get('lstm_confidence'),
                'lstm_probabilities': result.get('lstm_probabilities', {}),
                'buffer_size': result.get('buffer_size', 0),
                'final_alarm': final_alarm,
                'final_reason': final_reason,
                'final_source': final_source,
                'experiment_profile': self.experiment_profile,
                'fusion_enabled': bool(self.enable_fusion),
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
                'camera_id': self._camera_id,
                'camera_name': self._name,
                'status': self._status,
                'pipeline_available': bool(self.pipeline_available),
                'stream_fps': int(self.stream_fps) if self.stream_fps else 0,
                'infer_interval': float(self.infer_interval) if self.infer_interval else None,
                'experiment_profile': self.experiment_profile,
                'fusion_enabled': bool(self.enable_fusion),
                'last_detection': self._last_detection,
            }
    

