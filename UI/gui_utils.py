"""
GUI工具函数 - 线程安全的GUI更新方法
"""
import cv2
import tkinter as tk
import sys
import os
from PIL import Image, ImageTk

# 处理相对导入和绝对导入
def _import_config():
    """导入配置模块"""
    try:
        from .config import GUI_QUEUE_INTERVAL, MAX_INFO_LINES, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT
        return GUI_QUEUE_INTERVAL, MAX_INFO_LINES, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from hys.config import GUI_QUEUE_INTERVAL, MAX_INFO_LINES, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT
        return GUI_QUEUE_INTERVAL, MAX_INFO_LINES, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT


class ThreadSafeGUIUpdater:
    """线程安全的GUI更新器"""
    
    def __init__(self, root, gui_queue):
        self.root = root
        self.gui_queue = gui_queue
        self.frame_count = 0
        
    def process_gui_queue(self):
        """处理GUI更新队列（在主线程中调用）"""
        try:
            while True:
                try:
                    # 非阻塞获取队列中的任务
                    task = self.gui_queue.get_nowait()
                    if task is None:  # 停止信号
                        break
                    
                    task_type, args = task
                    
                    if task_type == 'update_status':
                        self._update_status_direct(args)
                    elif task_type == 'add_info':
                        self._add_info_direct(args)
                    elif task_type == 'update_frame':
                        self._update_frame_direct(args)
                    elif task_type == 'update_button_state':
                        self._update_button_state_direct(args)
                    
                except:
                    break  # 队列为空，退出循环
        except:
            pass  # 忽略错误，继续运行
        
        # 继续处理队列
        try:
            GUI_QUEUE_INTERVAL, _, _, _ = _import_config()
            self.root.after(GUI_QUEUE_INTERVAL, self.process_gui_queue)
        except:
            pass  # 窗口可能已关闭
    
    def _update_status_direct(self, args):
        """直接更新状态标签（仅在主线程调用）"""
        status_label, status = args
        try:
            if status_label and status_label.winfo_exists():
                status_label.config(text=f"状态: {status}")
        except:
            pass
    
    def _add_info_direct(self, args):
        """直接添加信息（仅在主线程调用）"""
        info_text, message = args
        try:
            if info_text and info_text.winfo_exists():
                info_text.insert(tk.END, f"{message}\n")
                info_text.see(tk.END)
                
                # 限制信息条数
                _, MAX_INFO_LINES, _, _ = _import_config()
                lines = info_text.get("1.0", tk.END).split('\n')
                if len(lines) > MAX_INFO_LINES:
                    info_text.delete("1.0", f"{len(lines) - MAX_INFO_LINES}.0")
        except:
            pass
    
    def _update_frame_direct(self, args):
        """直接更新帧（仅在主线程调用）"""
        video_label, frame, max_width, max_height = args
        try:
            if not video_label or not video_label.winfo_exists():
                return
            
            # 转换颜色空间 BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整大小以适应显示区域
            height, width = frame_rgb.shape[:2]
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # 转换为PIL Image
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)
            
            # 更新标签
            video_label.config(image=photo, text="")
            video_label.image = photo  # 保持引用
            
            # 更新帧计数
            self.frame_count += 1
        except:
            pass
    
    def _update_button_state_direct(self, args):
        """直接更新按钮状态（仅在主线程调用）"""
        buttons, button_name, state = args
        try:
            button = buttons.get(button_name)
            if button and button.winfo_exists():
                button.config(state=state)
        except:
            pass
    
    def update_status(self, status_label, status):
        """更新状态标签（线程安全）"""
        try:
            self.gui_queue.put(('update_status', [status_label, status]))
        except:
            pass
    
    def add_info(self, info_text, message):
        """添加信息到信息显示区域（线程安全）"""
        try:
            self.gui_queue.put(('add_info', [info_text, message]))
        except:
            pass
    
    def update_frame(self, video_label, frame):
        """更新显示的帧（线程安全）"""
        try:
            _, _, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT = _import_config()
            self.gui_queue.put(('update_frame', [video_label, frame, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT]))
        except:
            pass
    
    def update_button_state(self, buttons, button_name, state):
        """更新按钮状态（线程安全）"""
        try:
            self.gui_queue.put(('update_button_state', [buttons, button_name, state]))
        except:
            pass

