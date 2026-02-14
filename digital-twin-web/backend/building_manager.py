"""
建筑配置管理器 - 负责管理多建筑/楼层配置
"""
import json
import os
from pathlib import Path


class BuildingManager:
    """建筑配置管理器"""
    
    def __init__(self, config_path='../config/buildings.json'):
        """
        初始化建筑管理器
        
        Args:
            config_path: 建筑配置文件路径
        """
        self.config_path = Path(__file__).parent / config_path
        self.buildings = []
        self.current_building = None
        # 自动加载建筑配置
        self.load_buildings()
    
    def load_buildings(self):
        """加载所有建筑配置"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.buildings = data.get('buildings', [])
                    self.current_building = data.get('current_building', '')
                return True
            else:
                print(f"✗ 建筑配置文件不存在: {self.config_path}")
                self.buildings = []
                return False
        except Exception as e:
            print(f"✗ 建筑配置加载失败: {e}")
            self.buildings = []
            return False
    
    def get_current_building_id(self):
        """获取当前建筑ID"""
        return self.current_building
    
    def load_building_config(self, building_id):
        """
        从文件加载指定建筑的完整配置
        
        Args:
            building_id: 建筑ID
        
        Returns:
            建筑配置字典
        """
        # 查找建筑索引
        building_index = None
        for building in self.buildings:
            if building.get('id') == building_id:
                building_index = building
                break
        
        if not building_index:
            print(f"✗ 建筑不存在: {building_id}")
            return {}
        
        # 获取配置文件路径
        config_path = building_index.get('config_path')
        if not config_path:
            print(f"✗ 建筑配置路径未设置: {building_id}")
            return {}
        
        # 加载配置文件
        try:
            full_path = Path(__file__).parent / config_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"✗ 建筑配置文件不存在: {full_path}")
                return {}
        except Exception as e:
            print(f"✗ 建筑配置加载失败: {e}")
            return {}
    
    def get_building_list(self):
        """获取建筑列表（用于下拉框）"""
        building_list = []
        
        for building in self.buildings:
            building_info = {
                'id': building.get('id'),
                'name': building.get('name'),
                'floors': []
            }
            
            # 如果有楼层配置
            if 'floors' in building:
                building_info['floors'] = [
                    {
                        'id': floor.get('id'),
                        'name': floor.get('name')
                    }
                    for floor in building.get('floors', [])
                ]
            
            building_list.append(building_info)
        
        return building_list
    
    def get_building_config(self, building_id, floor_id=None):
        """
        获取指定建筑/楼层的配置
        
        Args:
            building_id: 建筑ID
            floor_id: 楼层ID（可选）
        
        Returns:
            建筑或楼层配置字典
        """
        for building in self.buildings:
            if building.get('id') == building_id:
                # 如果指定了楼层ID
                if floor_id and 'floors' in building:
                    for floor in building.get('floors', []):
                        if floor.get('id') == floor_id:
                            return floor
                    return None
                
                # 如果没有指定楼层ID
                if 'floors' in building and building['floors']:
                    # 返回第一个楼层
                    return building['floors'][0]
                else:
                    # 返回建筑本身
                    return building
        
        return None
    
    def switch_building(self, building_id, floor_id=None):
        """
        切换当前建筑/楼层
        
        Args:
            building_id: 建筑ID
            floor_id: 楼层ID（可选）
        
        Returns:
            切换后的配置
        """
        config = self.get_building_config(building_id, floor_id)
        if config:
            self.current_building = config
            print(f"✓ 切换到建筑: {building_id}" + (f" 楼层: {floor_id}" if floor_id else ""))
            return config
        else:
            print(f"✗ 建筑不存在: {building_id}")
            return None
    
    def get_current_building(self):
        """获取当前建筑配置"""
        return self.current_building
