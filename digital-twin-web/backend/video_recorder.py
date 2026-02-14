"""
视频录制器 - 负责保存告警视频快照
"""
import os
from collections import deque
from datetime import datetime
from pathlib import Path
import threading
import cv2
import time


class VideoRecorder:
    """视频录制器"""
    
    def __init__(self, video_dir='../data/videos'):
        """
        初始化视频录制器
        
        Args:
            video_dir: 视频存储目录
        """
        self.video_dir = Path(__file__).parent / video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # 录制开关
        self.recording_enabled = False
        
        # 每个摄像头的循环缓冲区（保存最近30秒）
        self.frame_buffers = {}
        self.buffer_size = 30 * 15  # 30秒 * 15fps
        
        # 告警标记
        self.alert_marks = {}
        
        # 缓冲区锁
        self.buffer_locks = {}
    
    def save_frame(self, camera_id, frame, detection_result):
        """
        保存帧到循环缓冲区
        
        Args:
            camera_id: 摄像头ID
            frame: 视频帧（numpy array）
            detection_result: 检测结果
        """
        # 检查录制开关
        if not self.recording_enabled:
            return
        
        try:
            # 初始化缓冲区
            if camera_id not in self.frame_buffers:
                self.frame_buffers[camera_id] = deque(maxlen=self.buffer_size)
                self.buffer_locks[camera_id] = threading.Lock()
            
            # 保存帧数据
            with self.buffer_locks[camera_id]:
                self.frame_buffers[camera_id].append({
                    'frame': frame.copy(),
                    'timestamp': datetime.now(),
                    'detection': detection_result
                })
        except Exception as e:
            print(f"保存帧失败: {e}")
    
    def mark_alert(self, camera_id, alert_time):
        """
        标记告警时间点并异步保存视频
        
        Args:
            camera_id: 摄像头ID
            alert_time: 告警时间
        """
        # 检查录制开关
        if not self.recording_enabled:
            return
        
        try:
            self.alert_marks[camera_id] = alert_time
            
            # 异步保存告警快照
            thread = threading.Thread(
                target=self._save_alert_snapshot,
                args=(camera_id, alert_time),
                daemon=True
            )
            thread.start()
        except Exception as e:
            print(f"标记告警失败: {e}")
    
    def _save_alert_snapshot(self, camera_id, alert_time):
        """
        保存告警前后30秒的视频
        
        Args:
            camera_id: 摄像头ID
            alert_time: 告警时间
        """
        try:
            # 等待一小段时间，确保缓冲区有足够的帧
            time.sleep(2)
            
            if camera_id not in self.frame_buffers:
                return
            
            # 获取告警前后的帧
            frames_to_save = []
            with self.buffer_locks.get(camera_id, threading.Lock()):
                for frame_data in self.frame_buffers[camera_id]:
                    time_diff = abs((frame_data['timestamp'] - alert_time).total_seconds())
                    if time_diff <= 30:  # 前后30秒
                        frames_to_save.append(frame_data)
            
            if not frames_to_save:
                print(f"没有可保存的帧: {camera_id}")
                return
            
            # 生成文件名
            filename = f"alert_{camera_id}_{alert_time.strftime('%Y%m%d_%H%M%S')}.mp4"
            filepath = self.video_dir / filename
            
            # 写入视频
            first_frame = frames_to_save[0]['frame']
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(filepath), fourcc, 15, (w, h))
            
            for frame_data in frames_to_save:
                writer.write(frame_data['frame'])
            
            writer.release()
            print(f"✓ 告警快照已保存: {filepath.name}")
            
        except Exception as e:
            print(f"保存告警快照失败: {e}")
    
    def get_playback_video(self, camera_id, start_time, duration_seconds=60):
        """
        获取历史回放视频（从保存的文件中读取）
        
        Args:
            camera_id: 摄像头ID
            start_time: 开始时间（ISO格式字符串）
            duration_seconds: 时长（秒）
            
        Returns:
            视频文件路径或None
        """
        try:
            # 解析时间
            start_dt = datetime.fromisoformat(start_time)
            
            # 查找匹配的告警视频文件
            pattern = f"alert_{camera_id}_{start_dt.strftime('%Y%m%d')}*.mp4"
            matching_files = list(self.video_dir.glob(pattern))
            
            if matching_files:
                # 返回最接近的文件
                return str(matching_files[0])
            
            return None
        except Exception as e:
            print(f"获取回放视频失败: {e}")
            return None
