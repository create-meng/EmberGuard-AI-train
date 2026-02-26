"""
传感器管理器 - 负责管理传感器数据和告警
"""
from datetime import datetime
import random
import threading
import time


class SensorManager:
    """传感器管理器"""
    
    def __init__(self, socketio=None, app=None):
        """Demo-only：只保留单机模拟与快照读取。"""
        self.sensors = {}
        self.sensor_data = {}
        self._lock = threading.Lock()
        self._base_values = {}
        self._running = False
        self._thread = None
    
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
        with self._lock:
            self.sensors[sensor_id] = {
                'id': sensor_id,
                'type': sensor_type,
                'name': name or sensor_id,
                'unit': unit or '',
                'threshold': threshold,
                'online': True
            }
    
    def _update_sensor_data(self, sensor_id, value):
        with self._lock:
            self.sensor_data[sensor_id] = {
                'value': value,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
    
    def get_all_sensors(self):
        """获取所有传感器数据"""
        result = []

        with self._lock:
            sensors = list(self.sensors.items())
            data_map = dict(self.sensor_data)

        for sensor_id, sensor in sensors:
            data = data_map.get(sensor_id, {})
            current_value = data.get('value', 0)
            threshold = sensor.get('threshold')

            status = 'normal'
            if sensor.get('online', False) and threshold is not None and current_value > threshold:
                status = 'alert'

            result.append({
                'id': sensor_id,
                'name': sensor.get('name', sensor_id),
                'type': sensor.get('type'),
                'unit': sensor.get('unit', ''),
                'current_value': current_value,
                'threshold': threshold,
                'status': status,
                'timestamp': data.get('timestamp'),
                'online': sensor.get('online', False)
            })
        
        return result
    
    def simulate_sensor_data(self):
        """演示模式：模拟传感器数据"""
        with self._lock:
            sensors = list(self.sensors.items())

        for sensor_id, sensor in sensors:
            if sensor_id not in self._base_values:
                if sensor['type'] == 'temperature_sensor':
                    self._base_values[sensor_id] = random.uniform(20, 35)
                elif sensor['type'] == 'humidity_sensor':
                    self._base_values[sensor_id] = random.uniform(45, 65)
                elif sensor['type'] == 'smoke_detector':
                    self._base_values[sensor_id] = random.uniform(5, 15)
                else:
                    self._base_values[sensor_id] = 20

            # 平滑漂移（无跳变）：随机游走 + 小概率持续偏移，用于“缓慢变异常”
            if sensor['type'] == 'humidity_sensor':
                change = random.uniform(-0.35, 0.35)
                drift = random.uniform(0.05, 0.18) if random.random() < 0.06 else 0.0
                change += drift
            else:
                change = random.uniform(-0.5, 0.5)
                drift = random.uniform(0.06, 0.22) if random.random() < 0.05 else 0.0
                change += drift
            self._base_values[sensor_id] += change

            # 温湿度取消“突刺跳变”；保留其他传感器的随机突刺
            if sensor['type'] in ('temperature_sensor', 'humidity_sensor'):
                if sensor['type'] == 'temperature_sensor':
                    # 允许缓慢升到告警阈值附近，但仍限制上界，避免不现实飙升
                    self._base_values[sensor_id] = max(15, min(72, self._base_values[sensor_id]))
                else:
                    self._base_values[sensor_id] = max(30, min(92, self._base_values[sensor_id]))
                value = self._base_values[sensor_id]
            else:
                spike_prob = 0.01
                if random.random() < spike_prob:
                    value = random.uniform(50, 80)
                else:
                    self._base_values[sensor_id] = max(0, min(40, self._base_values[sensor_id]))
                    value = self._base_values[sensor_id]

            self._update_sensor_data(sensor_id, round(value, 1))
    
    def start_simulation(self):
        """启动传感器数据模拟（演示模式）"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._thread.start()
    
    def _simulation_loop(self):
        """传感器数据模拟循环"""
        while self._running:
            try:
                self.simulate_sensor_data()
            except Exception as e:
                print(f"✗ 传感器模拟异常: {e}")

            time.sleep(1)
