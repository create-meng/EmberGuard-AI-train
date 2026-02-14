"""
配置管理器 - 负责加载、保存和验证系统配置
"""
import json
import os
from pathlib import Path


class ConfigManager:
    """系统配置管理器"""
    
    def __init__(self, config_path='../config/system_config.json'):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(__file__).parent / config_path
        self.config = {}
        # 自动加载配置
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                return True
            else:
                print(f"✗ 配置文件不存在: {self.config_path}")
                self.config = self._get_default_config()
                self.save_config(self.config)
                return False
        except Exception as e:
            print(f"✗ 配置加载失败: {e}")
            self.config = self._get_default_config()
            return False
    
    def save_config(self, config):
        """保存配置到文件"""
        try:
            # 确保目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            self.config = config
            print(f"✓ 配置保存成功")
            return True
        except Exception as e:
            print(f"✗ 配置保存失败: {e}")
            return False
    
    def get_config(self):
        """获取当前配置"""
        return self.config
    
    def validate_config(self):
        """验证配置文件格式"""
        required_keys = ['models', 'detection']
        
        for key in required_keys:
            if key not in self.config:
                print(f"✗ 配置验证失败: 缺少必需字段 '{key}'")
                return False
        
        print("✓ 配置验证通过")
        return True
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            "models": {
                "yolo": "../../runs/detect/train2/weights/best.pt",
                "lstm": "../../models/lstm/best.pt"
            },
            "detection": {
                "normal_fps": 5,
                "alert_fps": 15,
                "alert_cooldown": 5,
                "conf_threshold": 0.25
            }
        }
