"""
传感器管理器 - 负责管理传感器数据和告警
"""
import os
from datetime import datetime
import random
import threading
import time


class SensorManager:
    """传感器管理器"""
    
    def __init__(self, socketio=None, app=None):
        """初始化传感器管理器"""
        self.sensors = {}
        self.sensor_data = {}
        self.socketio = socketio
        self.app = app
        self.simulation_thread = None
        self.simulation_running = False
    
    def register_sensor(self, sensor_id, sensor_type, threshold=None, name=None, unit=None):
        """
        注册传感器
        
        Args:
            sensor_id: 传感器ID
            sensor_type: 传感器类型
            threshold: 告警阈值
            name: 传感器名称
            unit: 单位
        """
        self.sensors[sensor_id] = {
            'id': sensor_id,
            'type': sensor_type,
            'name': name or sensor_id,
            'unit': unit or '',
            'threshold': threshold,
            'status': 'online',
            'online': True
        }
    
    def update_sensor_data(self, sensor_id, value):
        """更新传感器数据"""
        self.sensor_data[sensor_id] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        # 检查阈值告警
        sensor = self.sensors.get(sensor_id)
        if sensor and sensor.get('threshold'):
            if value > sensor['threshold']:
                self._trigger_sensor_alert(sensor_id, value)
        
        # 通过WebSocket推送更新
        if self.socketio:
            sensor_info = self.sensors.get(sensor_id, {})
            event_data = {
                'sensor_id': sensor_id,
                'sensor_name': sensor_info.get('name', sensor_id),
                'sensor_type': sensor_info.get('type'),
                'value': value,
                'unit': sensor_info.get('unit', ''),
                'threshold': sensor_info.get('threshold'),
                'status': 'alert' if (sensor_info.get('threshold') and value > sensor_info.get('threshold')) else 'normal',
                'timestamp': self.sensor_data[sensor_id]['timestamp']
            }
            self.socketio.emit('sensor_update', event_data)
    
    def _trigger_sensor_alert(self, sensor_id, value):
        """触发传感器告警"""
        sensor = self.sensors.get(sensor_id, {})
        
        # 通过WebSocket推送告警
        if self.socketio:
            self.socketio.emit('sensor_alert', {
                'sensor_id': sensor_id,
                'sensor_name': sensor.get('name', sensor_id),
                'sensor_type': sensor.get('type'),
                'type': 'sensor_threshold',
                'value': value,
                'threshold': sensor.get('threshold'),
                'unit': sensor.get('unit', ''),
                'timestamp': datetime.now().isoformat()
            })
    
    def get_all_sensors(self):
        """获取所有传感器数据"""
        result = []
        
        for sensor_id, sensor in self.sensors.items():
            data = self.sensor_data.get(sensor_id, {})
            
            sensor_info = {
                'id': sensor_id,
                'name': sensor.get('name', sensor_id),
                'type': sensor.get('type'),
                'unit': sensor.get('unit', ''),
                'current_value': data.get('value', 0),
                'threshold': sensor.get('threshold'),
                'status': 'normal',
                'timestamp': data.get('timestamp'),
                'online': sensor.get('online', False)
            }
            
            # 判断状态
            if sensor_info['online'] and sensor_info['threshold']:
                if sensor_info['current_value'] > sensor_info['threshold']:
                    sensor_info['status'] = 'alert'
            
            result.append(sensor_info)
        
        return result
    
    def get_sensor_data(self, sensor_id):
        """获取传感器数据"""
        sensor = self.sensors.get(sensor_id)
        if not sensor:
            return None
        
        data = self.sensor_data.get(sensor_id, {})
        
        return {
            'id': sensor_id,
            'type': sensor.get('type'),
            'current_value': data.get('value', 0),
            'threshold': sensor.get('threshold'),
            'timestamp': data.get('timestamp'),
            'online': sensor.get('online', False)
        }
    
    def simulate_sensor_data(self):
        """演示模式：模拟传感器数据"""
        if not hasattr(self, '_base_values'):
            self._base_values = {}
            
        for sensor_id, sensor in self.sensors.items():
            if sensor_id not in self._base_values:
                # 初始随机一个合理数值
                if sensor['type'] == 'temperature_sensor':
                    self._base_values[sensor_id] = random.uniform(20, 35)
                elif sensor['type'] == 'smoke_detector':
                    self._base_values[sensor_id] = random.uniform(5, 15)
                else:
                    self._base_values[sensor_id] = 20
            
            # 每秒微调数值 (+/- 0.5)
            change = random.uniform(-0.5, 0.5)
            self._base_values[sensor_id] += change
            
            # 限制范围并偶尔触发告警 (1% 概率跳变到阈值以上)
            if random.random() < 0.01:
                if sensor['type'] == 'temperature_sensor':
                    value = random.uniform(60, 70)
                else:
                    value = random.uniform(50, 80)
            else:
                # 保持在合理范围
                if sensor['type'] == 'temperature_sensor':
                    self._base_values[sensor_id] = max(15, min(55, self._base_values[sensor_id]))
                else:
                    self._base_values[sensor_id] = max(0, min(40, self._base_values[sensor_id]))
                value = self._base_values[sensor_id]
            
            self.update_sensor_data(sensor_id, round(value, 1))
    
    def start_simulation(self):
        """启动传感器数据模拟（演示模式）"""
        # 如果已经有后台任务在运行，不要重复启动
        if hasattr(self, '_simulation_task') and self._simulation_task:
            self.simulation_running = True # 确保开关开启
            return
            
        self.simulation_running = True
        
        # 使用 socketio 的后台任务
        if self.socketio:
            self._simulation_task = self.socketio.start_background_task(self._simulation_loop)
        else:
            self.simulation_thread = threading.Thread(
                target=self._simulation_loop,
                daemon=True
            )
            self.simulation_thread.start()
    
    def stop_simulation(self):
        """停止传感器数据模拟"""
        self.simulation_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
    
    def _simulation_loop(self):
        """传感器数据模拟循环"""
        while self.simulation_running:
            try:
                self.simulate_sensor_data()
                self.socketio.sleep(1)  # 每1秒更新一次
            except Exception as e:
                print(f"✗ 传感器模拟异常: {e}")
                self.socketio.sleep(1) if self.socketio else time.sleep(1)
