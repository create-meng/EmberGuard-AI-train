"""
历史数据管理器 - 负责保存和查询历史检测数据
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path


class HistoryManager:
    """历史数据管理器"""
    
    def __init__(self, data_dir='../data/history'):
        """
        初始化历史数据管理器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(__file__).parent / data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = 60
    
    def save_detection_record(self, camera_id, detection_result):
        """
        保存检测记录
        
        Args:
            camera_id: 摄像头ID
            detection_result: 检测结果字典
        """
        try:
            # 创建记录
            record = {
                'timestamp': detection_result.get('timestamp', datetime.now().isoformat()),
                'lstm_prediction': detection_result.get('lstm_prediction'),
                'lstm_class_name': detection_result.get('lstm_class_name'),
                'lstm_confidence': detection_result.get('lstm_confidence'),
                'yolo_detections_count': len(detection_result.get('yolo_detections', [])),
                'has_detection': detection_result.get('has_detection', False)
            }
            
            # 按日期分文件存储
            date_str = datetime.now().strftime('%Y-%m-%d')
            file_path = self.data_dir / f"{camera_id}_{date_str}.json"
            
            # 读取现有记录
            records = []
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        records = json.load(f)
                except:
                    records = []
            
            # 追加新记录
            records.append(record)
            
            # 保存
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"保存检测记录失败: {e}")
    
    def get_history(self, camera_id, hours=12):
        """
        获取指定时间范围的历史数据
        
        Args:
            camera_id: 摄像头ID
            hours: 时间范围（小时）
            
        Returns:
            历史记录列表
        """
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            records = []
            
            # 读取相关日期的文件
            current_date = start_time.date()
            end_date = datetime.now().date()
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                file_path = self.data_dir / f"{camera_id}_{date_str}.json"
                
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            day_records = json.load(f)
                            # 过滤时间范围
                            for record in day_records:
                                try:
                                    record_time = datetime.fromisoformat(record['timestamp'])
                                    if record_time >= start_time:
                                        records.append(record)
                                except:
                                    continue
                    except:
                        pass
                
                current_date += timedelta(days=1)
            
            return records
        except Exception as e:
            print(f"获取历史数据失败: {e}")
            return []
    
    def cleanup_old_data(self):
        """清理超过60天的数据"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            for file_path in self.data_dir.glob('*.json'):
                try:
                    # 从文件名提取日期
                    filename = file_path.stem
                    date_str = filename.split('_')[-1]
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    if file_date.date() < cutoff_date.date():
                        file_path.unlink()  # 删除文件
                        print(f"清理旧数据: {file_path.name}")
                except:
                    continue
        except Exception as e:
            print(f"清理旧数据失败: {e}")
