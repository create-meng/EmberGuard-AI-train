"""
告警管理器 - 负责管理和存储告警事件
"""
import os
from datetime import datetime
import uuid


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        """初始化告警管理器"""
        self.alerts = []
        self.max_alerts = 100
    
    def add_alert(self, alert_data):
        """
        添加告警记录
        
        Args:
            alert_data: 告警数据字典，包含：
                - camera_id: 摄像头ID
                - camera_name: 摄像头名称（可选）
                - type: 告警类型 ('fire' or 'smoke')
                - confidence: 置信度
                - timestamp: 时间戳（可选）
                - lstm_class: LSTM分类结果（可选）
        
        Returns:
            告警记录字典
        """
        # 创建完整的告警记录
        alert = {
            'id': str(uuid.uuid4()),
            'camera_id': alert_data.get('camera_id'),
            'camera_name': alert_data.get('camera_name', alert_data.get('camera_id')),
            'type': alert_data.get('type'),
            'confidence': alert_data.get('confidence'),
            'timestamp': alert_data.get('timestamp', datetime.now().isoformat()),
            'lstm_class': alert_data.get('lstm_class')
        }
        
        # 插入到列表开头
        self.alerts.insert(0, alert)
        
        # 限制最大数量
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[:self.max_alerts]
        
        camera_name = alert['camera_name']
        alert_type = alert['type']
        confidence = alert['confidence']
        print(f"✓ 新告警: {camera_name} - {alert_type} ({confidence:.2%})")
        
        return alert
    
    def get_recent_alerts(self, limit=10):
        """
        获取最近的告警记录
        
        Args:
            limit: 返回数量限制
        
        Returns:
            告警列表
        """
        return self.alerts[:limit]
    
    def get_alert(self, alert_id):
        """
        获取指定告警详情
        
        Args:
            alert_id: 告警ID
        
        Returns:
            告警记录字典或None
        """
        for alert in self.alerts:
            if alert['id'] == alert_id:
                return alert
        return None
    
    def get_today_alert_count(self):
        """
        获取今日告警数
        
        Returns:
            今日告警数量
        """
        today = datetime.now().date()
        count = 0
        
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time.date() == today:
                count += 1
        
        return count
