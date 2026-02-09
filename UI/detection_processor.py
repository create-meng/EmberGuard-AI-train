"""
检测处理逻辑 - 屏幕、摄像头、文件检测
支持YOLO+LSTM双模型架构
"""
import threading
import tkinter as tk
import sys
import os
from pathlib import Path
from ultralytics import YOLO

# 处理相对导入和绝对导入
try:
    from .file_handler import FileHandler
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hys.file_handler import FileHandler

# 尝试导入LSTM模块
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from emberguard.pipeline import FireDetectionPipeline
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️  LSTM模块未找到，将只使用YOLO检测")


class DetectionProcessor:
    """检测处理器"""
    
    def __init__(self, yolo_model, gui_updater, buttons, status_label, info_text, video_label, 
                 use_lstm=False, lstm_model_path=None):
        self.yolo = yolo_model
        self.gui_updater = gui_updater
        self.buttons = buttons
        self.status_label = status_label
        self.info_text = info_text
        self.video_label = video_label
        self.is_running = False
        self.frame_count = 0
        self.conf = 0.25  # 默认置信度阈值
        self.iou = 0.45   # 默认IoU阈值
        self.save_dir = None  # 保存文件夹
        self.save_frame_count = 0  # 保存的帧计数
        
        # LSTM相关
        self.use_lstm = use_lstm and LSTM_AVAILABLE
        self.lstm_pipeline = None
        
        # 如果启用LSTM，初始化管道
        if self.use_lstm:
            try:
                # 获取YOLO模型路径
                yolo_path = yolo_model.ckpt_path if hasattr(yolo_model, 'ckpt_path') else 'runs/detect/train2/weights/best.pt'
                
                # 使用默认LSTM模型路径或指定路径
                if lstm_model_path is None:
                    lstm_model_path = 'models/lstm/train/best.pt'
                
                # 检查LSTM模型是否存在
                if Path(lstm_model_path).exists():
                    self.lstm_pipeline = FireDetectionPipeline(
                        yolo_model_path=yolo_path,
                        lstm_model_path=lstm_model_path,
                        sequence_length=30
                    )
                    self.gui_updater.add_info(self.info_text, "✅ LSTM模型已加载")
                else:
                    self.use_lstm = False
                    self.gui_updater.add_info(self.info_text, f"⚠️  LSTM模型不存在: {lstm_model_path}")
                    self.gui_updater.add_info(self.info_text, "将只使用YOLO检测")
            except Exception as e:
                self.use_lstm = False
                self.gui_updater.add_info(self.info_text, f"⚠️  LSTM加载失败: {e}")
                self.gui_updater.add_info(self.info_text, "将只使用YOLO检测")
    
    def set_params(self, conf, iou):
        """设置检测参数"""
        self.conf = conf
        self.iou = iou
    
    def set_save_dir(self, save_dir):
        """设置保存文件夹"""
        self.save_dir = save_dir
        # 确保文件夹存在
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def save_detected_frame(self, result):
        """保存检测到目标的帧"""
        if not self.save_dir or len(result.boxes) == 0:
            return
        
        try:
            import cv2
            import os
            import numpy as np
            from datetime import datetime
            
            # 获取原始图像
            if hasattr(result, 'orig_img'):
                orig_img = result.orig_img
                # YOLO返回的orig_img通常是RGB格式，需要转为BGR
                if isinstance(orig_img, np.ndarray) and len(orig_img.shape) == 3:
                    # 检查是否是RGB格式（通常YOLO返回RGB）
                    # 转换为BGR格式用于OpenCV保存
                    orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
                else:
                    orig_img_bgr = orig_img
                
                # 生成文件名（时间戳 + 帧编号）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detected_{timestamp}_{self.save_frame_count:06d}.jpg"
                filepath = os.path.join(self.save_dir, filename)
                
                # 保存帧
                cv2.imwrite(filepath, orig_img_bgr)
                self.save_frame_count += 1
                
                # 每保存10帧提示一次（避免信息过多）
                if self.save_frame_count % 10 == 0:
                    self.gui_updater.add_info(self.info_text, 
                                             f"已保存 {self.save_frame_count} 帧到: {os.path.basename(self.save_dir)}")
        except Exception as e:
            # 保存失败时不中断检测，只记录错误
            pass
    
    def start_screen_detection(self):
        """启动屏幕检测"""
        if self.is_running:
            return
        
        try:
            self.is_running = True
            self.buttons['start'].config(state=tk.DISABLED)
            self.buttons['stop'].config(state=tk.NORMAL)
            self.gui_updater.update_status(self.status_label, "屏幕检测运行中...")
            self.gui_updater.add_info(self.info_text, "开始屏幕检测...")
            
            # 在新线程中运行屏幕检测
            detection_thread = threading.Thread(target=self.process_screen, daemon=True)
            detection_thread.start()
            
        except Exception as e:
            self.gui_updater.update_status(self.status_label, f"错误: {str(e)}")
            self.gui_updater.add_info(self.info_text, f"启动屏幕检测时发生错误: {str(e)}")
            self.is_running = False
    
    def start_camera_detection(self):
        """启动摄像头检测"""
        if self.is_running:
            return
        
        try:
            self.is_running = True
            self.buttons['start'].config(state=tk.DISABLED)
            self.buttons['stop'].config(state=tk.NORMAL)
            self.gui_updater.update_status(self.status_label, "摄像头检测运行中...")
            self.gui_updater.add_info(self.info_text, "摄像头已启动，开始实时检测...")
            
            # 在新线程中运行视频处理（YOLO会自动处理摄像头）
            detection_thread = threading.Thread(target=self.process_camera, daemon=True)
            detection_thread.start()
            
        except Exception as e:
            self.gui_updater.update_status(self.status_label, f"错误: {str(e)}")
            self.gui_updater.add_info(self.info_text, f"启动摄像头时发生错误: {str(e)}")
            self.is_running = False
    
    def start_file_detection(self, file_path, detection_file_type_callback, 
                            detection_results_callback, detection_has_results_callback):
        """开始文件检测"""
        if self.is_running:
            return
        
        try:
            self.gui_updater.update_status(self.status_label, "处理中...")
            self.gui_updater.add_info(self.info_text, f"正在处理文件: {file_path}")
            
            # 检查文件类型
            is_video = FileHandler.is_video_file(file_path)
            
            if is_video:
                # 视频文件：使用stream模式处理
                detection_file_type_callback('video')
                self.is_running = True
                self.buttons['start'].config(state=tk.DISABLED)
                self.buttons['stop'].config(state=tk.NORMAL)
                if 'save' in self.buttons:
                    self.buttons['save'].config(state=tk.DISABLED)
                
                # 在新线程中处理视频
                detection_thread = threading.Thread(
                    target=self.process_video_file, 
                    args=(file_path, detection_results_callback, detection_has_results_callback), 
                    daemon=True
                )
                detection_thread.start()
            else:
                # 图片文件：直接处理
                detection_file_type_callback('image')
                
                # 使用LSTM管道或纯YOLO
                if self.use_lstm and self.lstm_pipeline:
                    # 读取图片
                    import cv2
                    frame = cv2.imread(file_path)
                    
                    # 重置LSTM缓冲区
                    self.lstm_pipeline.reset_buffer()
                    
                    # 填充缓冲区（重复30次）
                    lstm_result = None
                    for _ in range(30):
                        lstm_result = self.lstm_pipeline.detect_frame(frame, conf_threshold=self.conf)
                    
                    # 绘制结果
                    annotated_frame = self.lstm_pipeline._draw_results(frame, lstm_result)
                    
                    # 保存检测结果
                    detection_results_callback(annotated_frame, [lstm_result])
                    
                    # 更新显示
                    self.gui_updater.update_frame(self.video_label, annotated_frame)
                    
                    # 检查是否有检测结果
                    has_detections = lstm_result['has_detection']
                    detection_has_results_callback(has_detections)
                    
                    # 显示检测信息
                    if has_detections or 'lstm_prediction' in lstm_result:
                        # 构造YOLO results格式用于显示
                        class DummyResult:
                            def __init__(self, detections):
                                self.boxes = detections
                        
                        dummy_boxes = lstm_result['yolo_detections'] if lstm_result['yolo_detections'] else []
                        dummy_result = DummyResult(dummy_boxes)
                        
                        self.update_detection_info([dummy_result], show_all=True, lstm_result=lstm_result)
                        self.gui_updater.add_info(self.info_text, "检测完成，可以点击'保存检测结果'保存")
                    else:
                        self.gui_updater.add_info(self.info_text, "未检测到目标")
                        self.gui_updater.add_info(self.info_text, "检测完成，可以点击'保存检测结果'保存")
                else:
                    # 纯YOLO检测
                    results = self.yolo(file_path, verbose=False, conf=self.conf, iou=self.iou)
                    
                    # 绘制检测结果
                    annotated_frame = results[0].plot()
                    
                    # 保存检测结果
                    detection_results_callback(annotated_frame, results)
                    
                    # 更新显示
                    self.gui_updater.update_frame(self.video_label, annotated_frame)
                    
                    # 检查是否有检测结果
                    has_detections = len(results[0].boxes) > 0
                    detection_has_results_callback(has_detections)
                    
                    # 显示检测信息
                    if has_detections:
                        self.update_detection_info(results, show_all=True)
                        self.gui_updater.add_info(self.info_text, "检测完成，可以点击'保存检测结果'保存")
                    else:
                        self.gui_updater.add_info(self.info_text, "未检测到目标")
                        self.gui_updater.add_info(self.info_text, "检测完成，可以点击'保存检测结果'保存")
                
                # 启用保存按钮
                if 'save' in self.buttons:
                    self.gui_updater.update_button_state(self.buttons, 'save', tk.NORMAL)
                self.gui_updater.update_status(self.status_label, "检测完成")
                self.is_running = False
                self.gui_updater.update_button_state(self.buttons, 'start', tk.NORMAL)
                self.gui_updater.update_button_state(self.buttons, 'stop', tk.DISABLED)
                
        except Exception as e:
            self.gui_updater.update_status(self.status_label, f"错误: {str(e)}")
            self.gui_updater.add_info(self.info_text, f"处理文件时发生错误: {str(e)}")
            from tkinter import messagebox
            messagebox.showerror("错误", f"处理文件时发生错误: {str(e)}")
            self.is_running = False
            self.gui_updater.update_button_state(self.buttons, 'start', tk.NORMAL)
            self.gui_updater.update_button_state(self.buttons, 'stop', tk.DISABLED)
            if 'save' in self.buttons:
                self.gui_updater.update_button_state(self.buttons, 'save', tk.DISABLED)
    
    def process_screen(self):
        """处理屏幕检测"""
        try:
            # source="screen" 会自动处理屏幕捕获
            for result in self.yolo(source="screen", stream=True, verbose=False, 
                                   conf=self.conf, iou=self.iou):
                if not self.is_running:
                    break
                
                # 获取带标注的帧
                annotated_frame = result.plot()
                
                # 如果检测到目标，保存原始帧（不带标注）
                if len(result.boxes) > 0:
                    self.save_detected_frame(result)
                
                # 更新显示
                self.gui_updater.update_frame(self.video_label, annotated_frame)
                
                # 显示检测信息
                self.update_detection_info([result])
                
        except Exception as e:
            self.gui_updater.add_info(self.info_text, f"屏幕检测时发生错误: {str(e)}")
            self.is_running = False
            self.gui_updater.update_button_state(self.buttons, 'start', tk.NORMAL)
            self.gui_updater.update_button_state(self.buttons, 'stop', tk.DISABLED)
    
    def process_camera(self):
        """处理摄像头检测"""
        try:
            if self.use_lstm and self.lstm_pipeline:
                # 使用LSTM管道
                import cv2
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    self.gui_updater.add_info(self.info_text, "❌ 无法打开摄像头")
                    return
                
                # 重置LSTM缓冲区
                self.lstm_pipeline.reset_buffer()
                
                while self.is_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # LSTM检测
                    lstm_result = self.lstm_pipeline.detect_frame(frame, conf_threshold=self.conf)
                    
                    # 绘制结果
                    annotated_frame = self.lstm_pipeline._draw_results(frame, lstm_result)
                    
                    # 如果检测到目标，保存原始帧
                    if lstm_result['has_detection']:
                        # 构造YOLO result格式用于保存
                        class DummyResult:
                            def __init__(self, frame, detections):
                                self.orig_img = frame
                                self.boxes = detections
                        dummy_result = DummyResult(frame, lstm_result['yolo_detections'])
                        self.save_detected_frame(dummy_result)
                    
                    # 更新显示
                    self.gui_updater.update_frame(self.video_label, annotated_frame)
                    
                    # 显示检测信息
                    class DummyResult:
                        def __init__(self, detections):
                            self.boxes = detections
                    dummy_boxes = lstm_result['yolo_detections'] if lstm_result['yolo_detections'] else []
                    dummy_result = DummyResult(dummy_boxes)
                    self.update_detection_info([dummy_result], lstm_result=lstm_result)
                
                cap.release()
            else:
                # 纯YOLO检测
                for result in self.yolo(source=0, stream=True, verbose=False,
                                       conf=self.conf, iou=self.iou):
                    if not self.is_running:
                        break
                    
                    # 获取带标注的帧
                    annotated_frame = result.plot()
                    
                    # 如果检测到目标，保存原始帧（不带标注）
                    if len(result.boxes) > 0:
                        self.save_detected_frame(result)
                    
                    # 更新显示
                    self.gui_updater.update_frame(self.video_label, annotated_frame)
                    
                    # 显示检测信息
                    self.update_detection_info([result])
        except Exception as e:
            self.gui_updater.add_info(self.info_text, f"摄像头检测时发生错误: {str(e)}")
        finally:
            # 清理资源：关闭 dataset 以释放摄像头
            if hasattr(self.yolo, 'predictor') and self.yolo.predictor is not None:
                predictor = self.yolo.predictor
                if hasattr(predictor, 'dataset') and predictor.dataset is not None:
                    dataset = predictor.dataset
                    if hasattr(dataset, 'close'):
                        dataset.close()
            self.is_running = False
            self.gui_updater.update_button_state(self.buttons, 'start', tk.NORMAL)
            self.gui_updater.update_button_state(self.buttons, 'stop', tk.DISABLED)
    
    def process_video_file(self, file_path, detection_results_callback, detection_has_results_callback):
        """处理视频文件"""
        try:
            # 使用YOLO的stream模式处理视频
            has_detections = False
            video_frames = []  # 保存所有处理后的帧
            
            for result in self.yolo(source=file_path, stream=True, verbose=False,
                                   conf=self.conf, iou=self.iou):
                if not self.is_running:
                    break
                
                # 检查是否有检测结果
                if len(result.boxes) > 0:
                    has_detections = True
                
                # 获取带标注的帧
                annotated_frame = result.plot()
                
                # 保存帧到列表
                video_frames.append(annotated_frame.copy())
                
                # 更新显示
                self.gui_updater.update_frame(self.video_label, annotated_frame)
                
                # 显示检测信息
                self.update_detection_info([result])
            
            if self.is_running:
                # 保存检测结果
                detection_results_callback(video_frames, None)
                detection_has_results_callback(has_detections)
                
                if has_detections:
                    self.gui_updater.add_info(self.info_text, "检测完成，可以点击'保存检测结果'保存")
                else:
                    self.gui_updater.add_info(self.info_text, "未检测到目标")
                    self.gui_updater.add_info(self.info_text, "检测完成，可以点击'保存检测结果'保存")
                
                # 启用保存按钮
                if 'save' in self.buttons:
                    self.gui_updater.update_button_state(self.buttons, 'save', tk.NORMAL)
                self.gui_updater.add_info(self.info_text, "视频处理完成")
                self.gui_updater.update_status(self.status_label, "检测完成")
                self.is_running = False
                self.gui_updater.update_button_state(self.buttons, 'start', tk.NORMAL)
                self.gui_updater.update_button_state(self.buttons, 'stop', tk.DISABLED)
                
        except Exception as e:
            self.gui_updater.add_info(self.info_text, f"处理视频时发生错误: {str(e)}")
            self.is_running = False
            self.gui_updater.update_button_state(self.buttons, 'start', tk.NORMAL)
            self.gui_updater.update_button_state(self.buttons, 'stop', tk.DISABLED)
            if 'save' in self.buttons:
                self.gui_updater.update_button_state(self.buttons, 'save', tk.DISABLED)
    
    def process_frame_with_lstm(self, frame):
        """
        使用LSTM处理单帧
        
        Args:
            frame: 输入帧
            
        Returns:
            dict: 检测结果
        """
        if not self.use_lstm or self.lstm_pipeline is None:
            return None
        
        try:
            result = self.lstm_pipeline.detect_frame(frame, conf_threshold=self.conf)
            return result
        except Exception as e:
            # LSTM检测失败时不中断，只记录错误
            if self.frame_count % 100 == 0:  # 每100帧提示一次
                self.gui_updater.add_info(self.info_text, f"LSTM检测失败: {e}")
            return None
    
    def update_detection_info(self, results, show_all=False, lstm_result=None):
        """更新检测信息"""
        # YOLO检测信息
        if len(results[0].boxes) > 0:
            detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.yolo.names[cls_id]
                detections.append(f"{cls_name}: {conf:.2f}")
            
            # 对于图片检测，显示所有信息；对于实时检测，每10帧更新一次
            if show_all or self.frame_count % 10 == 0:
                info = f"🔍 YOLO: 检测到 {len(results[0].boxes)} 个目标: {', '.join(detections[:10])}"
                if len(detections) > 10:
                    info += f" ... (共{len(detections)}个)"
                self.gui_updater.add_info(self.info_text, info)
        elif show_all:
            # 文件检测模式下，如果没有检测到目标，显示提示
            self.gui_updater.add_info(self.info_text, "🔍 YOLO: 未检测到目标")
        
        # LSTM预测信息
        if lstm_result and 'lstm_prediction' in lstm_result:
            if show_all or self.frame_count % 10 == 0:
                class_name = lstm_result['lstm_class_name']
                confidence = lstm_result['lstm_confidence']
                info = f"🧠 LSTM: {class_name} (置信度: {confidence:.3f})"
                self.gui_updater.add_info(self.info_text, info)
                
                # 显示概率分布（仅在show_all时）
                if show_all:
                    probs = lstm_result['lstm_probabilities']
                    prob_info = f"   概率分布: 无火={probs['无火']:.3f}, 烟雾={probs['烟雾']:.3f}, 火焰={probs['火焰']:.3f}"
                    self.gui_updater.add_info(self.info_text, prob_info)
        elif self.use_lstm and lstm_result and show_all:
            # LSTM缓冲区未满
            buffer_size = lstm_result.get('buffer_size', 0)
            self.gui_updater.add_info(self.info_text, f"🧠 LSTM: 缓冲区填充中 ({buffer_size}/30)")
        
        self.frame_count += 1
    
    def update_detection_info_old(self, results, show_all=False):
        """更新检测信息"""
        if len(results[0].boxes) > 0:
            detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.yolo.names[cls_id]
                detections.append(f"{cls_name}: {conf:.2f}")
            
            # 对于图片检测，显示所有信息；对于实时检测，每10帧更新一次
            if show_all or self.frame_count % 10 == 0:
                info = f"检测到 {len(results[0].boxes)} 个目标: {', '.join(detections[:10])}"
                if len(detections) > 10:
                    info += f" ... (共{len(detections)}个)"
                self.gui_updater.add_info(self.info_text, info)
        elif show_all:
            # 文件检测模式下，如果没有检测到目标，显示提示
            self.gui_updater.add_info(self.info_text, "未检测到目标")
        
        self.frame_count += 1
    
    def stop(self):
        """停止检测"""
        self.is_running = False

