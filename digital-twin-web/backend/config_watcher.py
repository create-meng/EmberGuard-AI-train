"""
配置文件监听器 - 监听配置文件变化并通知前端
"""
import os
import time
from queue import Queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变化处理器"""
    
    def __init__(self, event_queue):
        self.event_queue = event_queue
        self.last_modified = {}
        self.debounce_time = 1.0  # 防抖时间（秒）
    
    def on_modified(self, event):
        """文件修改事件"""
        if event.is_directory:
            return
        
        # 只监听 .json 配置文件
        if not event.src_path.endswith('.json'):
            return
        
        # 防抖：避免短时间内多次触发
        current_time = time.time()
        last_time = self.last_modified.get(event.src_path, 0)
        
        if current_time - last_time < self.debounce_time:
            return
        
        self.last_modified[event.src_path] = current_time
        
        # 获取文件名
        filename = os.path.basename(event.src_path)
        
        # 将事件放入队列，由 socketio 后台任务处理
        self.event_queue.put({
            'file': filename,
            'path': event.src_path,
            'timestamp': current_time
        })


class ConfigWatcher:
    """配置文件监听器"""
    
    def __init__(self, socketio, building_manager, app):
        self.socketio = socketio
        self.building_manager = building_manager
        self.app = app
        self.observer = None
        self.handler = None
        self.event_queue = Queue()
        self.running = False
    
    def start(self):
        """启动监听"""
        # 获取配置文件目录
        config_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..',
            'config'
        ))
        
        buildings_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..',
            'buildings'
        ))
        
        # 创建事件处理器（传入队列而不是socketio）
        self.handler = ConfigFileHandler(self.event_queue)
        
        # 创建观察者
        self.observer = Observer()
        
        # 监听配置目录
        if os.path.exists(config_dir):
            self.observer.schedule(self.handler, config_dir, recursive=False)
        
        # 监听建筑目录
        if os.path.exists(buildings_dir):
            self.observer.schedule(self.handler, buildings_dir, recursive=True)
        
        # 启动观察者
        self.observer.start()
        
        # 启动 socketio 后台任务来处理队列中的事件
        self.running = True
        self.socketio.start_background_task(self._emit_loop)
        
        print("✓ 配置文件监听已启动")
    
    def _emit_loop(self):
        """Socket.IO 后台任务：从队列读取事件并发送"""
        while self.running:
            try:
                # 非阻塞地检查队列
                if not self.event_queue.empty():
                    event_data = self.event_queue.get_nowait()
                    
                    # 立即触发重新加载建筑配置，确保后端数据是最新的
                    if 'buildings.json' in event_data['file'] or event_data['file'].endswith('_config.json'):
                        print(f"DEBUG: 检测到配置文件 {event_data['file']} 变化，正在重新加载...")
                        self.building_manager.load_buildings()
                    
                    # 在 socketio 上下文中发送事件
                    self.socketio.emit('config_changed', event_data)
                    print(f"✓ 配置已更新: {event_data['file']}")
                
                # 短暂休眠，避免占用过多CPU
                self.socketio.sleep(0.1)
            except Exception as e:
                self.socketio.sleep(0.1)
    
    def stop(self):
        """停止监听"""
        self.running = False
        if self.observer:
            self.observer.stop()
            self.observer.join()
