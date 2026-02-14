"""
统一API路由管理器
"""
import os
from flask import jsonify, request, send_file, render_template
from datetime import datetime
import os


class APIRouter:
    """统一的API路由管理器"""
    
    def __init__(self, app, config_manager, building_manager, detection_engine, 
                 alert_manager, sensor_manager, history_manager, video_recorder):
        """
        初始化API路由管理器
        
        Args:
            app: Flask应用实例
            config_manager: 配置管理器
            building_manager: 建筑管理器
            detection_engine: 检测引擎
            alert_manager: 告警管理器
            sensor_manager: 传感器管理器
            history_manager: 历史数据管理器
            video_recorder: 视频录制器
        """
        self.app = app
        self.config_manager = config_manager
        self.building_manager = building_manager
        self.detection_engine = detection_engine
        self.alert_manager = alert_manager
        self.sensor_manager = sensor_manager
        self.history_manager = history_manager
        self.video_recorder = video_recorder
        
        self.register_routes()
    
    def register_routes(self):
        """注册所有路由"""
        # 页面路由
        self.app.route('/')(self.index)
        self.app.route('/api/config', methods=['GET'])(self.get_config)
        self.app.route('/api/config', methods=['POST'])(self.update_config)
        
        # 建筑静态文件路由
        self.app.route('/buildings/<path:filepath>')(self.serve_building_file)
        
        # 建筑相关API
        self.app.route('/api/buildings', methods=['GET'])(self.get_buildings)
        self.app.route('/api/buildings/<building_id>', methods=['GET'])(self.get_building)
        self.app.route('/api/buildings/switch', methods=['POST'])(self.switch_building)
        self.app.route('/api/buildings/clear', methods=['POST'])(self.clear_building)
        self.app.route('/api/buildings/<building_id>/floors/<floor_id>/switch', methods=['POST'])(self.switch_floor)
        
        # 摄像头相关API
        self.app.route('/api/cameras', methods=['GET'])(self.get_cameras)
        self.app.route('/api/cameras/<camera_id>', methods=['GET'])(self.get_camera)
        self.app.route('/api/cameras/<camera_id>/status', methods=['GET'])(self.get_camera_status)
        
        # 传感器相关API
        self.app.route('/api/sensors', methods=['GET'])(self.get_sensors)
        self.app.route('/api/sensors/<sensor_id>', methods=['GET'])(self.get_sensor)
        
        # 告警相关API
        self.app.route('/api/alerts', methods=['GET'])(self.get_alerts)
        self.app.route('/api/alerts/<alert_id>', methods=['GET'])(self.get_alert)
        
        # 历史数据API
        self.app.route('/api/history/<camera_id>', methods=['GET'])(self.get_history)
        self.app.route('/api/history/<camera_id>/video', methods=['GET'])(self.get_history_video)
        
        # 系统状态API
        self.app.route('/api/system/status', methods=['GET'])(self.get_system_status)
        self.app.route('/api/system/stop-all', methods=['POST'])(self.stop_all_devices)
        self.app.route('/api/system/settings', methods=['GET'])(self.get_system_settings)
        self.app.route('/api/system/settings', methods=['POST'])(self.update_system_settings)
        
        if not os.environ.get('SILENT_MODE'):
            print("✓ API路由注册完成")
    
    # ========== 页面路由 ==========
    def index(self):
        """主页"""
        from flask import render_template
        try:
            return render_template('index.html')
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'前端页面未找到: {str(e)}'
            }), 404
    
    def serve_building_file(self, filepath):
        """提供建筑文件夹中的静态文件（如平面图）"""
        import os
        from flask import send_from_directory
        
        buildings_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '..',
            'buildings'
        ))
        
        try:
            return send_from_directory(buildings_dir, filepath)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'文件不存在: {filepath}'
            }), 404
    
    # ========== 配置API ==========
    def get_config(self):
        """获取系统配置"""
        try:
            config = self.config_manager.get_config()
            return jsonify({
                'success': True,
                'data': config
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def update_config(self):
        """更新系统配置"""
        try:
            data = request.json
            success = self.config_manager.save_config(data)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': '配置更新成功'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '配置保存失败'
                }), 500
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # ========== 建筑API ==========
    def get_buildings(self):
        """获取所有建筑列表"""
        try:
            buildings = self.building_manager.get_building_list()
            return jsonify({
                'success': True,
                'data': buildings
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def get_building(self, building_id):
        """获取指定建筑配置"""
        try:
            # 使用 load_building_config 加载完整配置
            building = self.building_manager.load_building_config(building_id)
            
            if building:
                return jsonify({
                    'success': True,
                    'data': building
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '建筑不存在'
                }), 404
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def switch_building(self):
        """切换当前建筑"""
        try:
            data = request.json
            building_id = data.get('building_id')
            
            if not building_id:
                return jsonify({
                    'success': False,
                    'error': '缺少 building_id 参数'
                }), 400
            
            # 加载建筑配置
            building_config = self.building_manager.load_building_config(building_id)
            
            if not building_config:
                return jsonify({
                    'success': False,
                    'error': f'建筑配置文件不存在或加载失败: {building_id}'
                }), 404
            
            # 停止所有当前摄像头
            for camera_id in list(self.detection_engine.cameras.keys()):
                self.detection_engine.stop_camera(camera_id)
            
            # 停止传感器模拟
            self.sensor_manager.stop_simulation()
            
            # 清空传感器
            self.sensor_manager.sensors.clear()
            self.sensor_manager.sensor_data.clear()
            
            # 获取新建筑的所有设施（包含所有楼层）
            facilities = []
            if 'floors' in building_config:
                # 合并所有楼层的设施
                for floor in building_config.get('floors', []):
                    facilities.extend(floor.get('facilities', []))
            else:
                # 兼容旧格式
                facilities = building_config.get('facilities', [])
            
            # 注册所有楼层的传感器
            sensor_count = 0
            for facility in facilities:
                if facility.get('type') in ['temperature_sensor', 'humidity_sensor', 'smoke_detector']:
                    self.sensor_manager.register_sensor(
                        sensor_id=facility['id'],
                        sensor_type=facility['type'],
                        threshold=facility.get('threshold'),
                        name=facility.get('name'),
                        unit=facility.get('unit')
                    )
                    sensor_count += 1
            
            # 启动所有楼层的摄像头
            camera_configs = [f for f in facilities if f.get('type') == 'camera']
            
            # 自动检测：如果任何摄像头有 demo_video 字段，就启动传感器模拟
            has_demo_video = any(cam.get('demo_video') for cam in camera_configs)
            
            if camera_configs:
                self.detection_engine.start_all_cameras(camera_configs, has_demo_video)
            
            # 如果有传感器且有演示视频，启动传感器模拟
            if sensor_count > 0 and has_demo_video:
                self.sensor_manager.start_simulation()
            
            return jsonify({
                'success': True,
                'message': f'已切换到建筑: {building_config.get("name", building_id)}',
                'data': {
                    'building_id': building_id,
                    'facilities_count': len(facilities),
                    'cameras_count': len(camera_configs)
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'切换建筑失败: {str(e)}'
            }), 500
    
    def clear_building(self):
        """清空当前建筑选择（页面刷新时调用）"""
        try:
            # 停止所有摄像头
            for camera_id in list(self.detection_engine.cameras.keys()):
                self.detection_engine.stop_camera(camera_id)
            
            # 停止传感器模拟
            self.sensor_manager.stop_simulation()
            
            # 清空传感器
            self.sensor_manager.sensors.clear()
            self.sensor_manager.sensor_data.clear()
            
            return jsonify({
                'success': True,
                'message': '已清空建筑选择'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def switch_floor(self, building_id, floor_id):
        """切换楼层（仅切换前端显示，不停止设备）"""
        print(f"DEBUG: 收到楼层切换请求 - Building: {building_id}, Floor: {floor_id}")
        try:
            # 加载建筑配置
            building_config = self.building_manager.load_building_config(building_id)
            
            if not building_config:
                return jsonify({
                    'success': False,
                    'error': f'建筑配置不存在: {building_id}'
                }), 404
            
            # 查找指定楼层
            floors = building_config.get('floors', [])
            target_floor = None
            for floor in floors:
                if floor['id'] == floor_id:
                    target_floor = floor
                    break
            
            if not target_floor:
                return jsonify({
                    'success': False,
                    'error': f'楼层不存在: {floor_id}'
                }), 404
            
            # 注意：不停止任何设备，所有楼层的设备继续运行
            # 只返回当前楼层的设施信息供前端显示
            
            facilities = target_floor.get('facilities', [])
            camera_count = len([f for f in facilities if f.get('type') == 'camera'])
            sensor_count = len([f for f in facilities if f.get('type') in ['temperature_sensor', 'humidity_sensor', 'smoke_detector']])
            
            print(f"DEBUG: 楼层切换成功 - {target_floor.get('name', floor_id)}")
            return jsonify({
                'success': True,
                'message': f'已切换到 {target_floor.get("name", floor_id)}（所有楼层设备继续运行）',
                'data': {
                    'floor_id': floor_id,
                    'floor_name': target_floor.get('name'),
                    'cameras': camera_count,
                    'sensors': sensor_count
                }
            })
        except Exception as e:
            print(f"ERROR: 楼层切换异常 - {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # ========== 摄像头API ==========
    def get_cameras(self):
        """获取所有摄像头状态"""
        try:
            cameras = self.detection_engine.get_all_camera_status()
            return jsonify({
                'success': True,
                'data': cameras
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def get_camera(self, camera_id):
        """获取指定摄像头信息"""
        try:
            camera = self.detection_engine.get_camera_info(camera_id)
            
            if camera:
                return jsonify({
                    'success': True,
                    'data': camera
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '摄像头不存在'
                }), 404
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def get_camera_status(self, camera_id):
        """获取摄像头实时状态"""
        try:
            status = self.detection_engine.get_camera_status(camera_id)
            
            if status:
                return jsonify({
                    'success': True,
                    'data': status
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '摄像头不存在'
                }), 404
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # ========== 传感器API ==========
    def get_sensors(self):
        """获取所有传感器数据"""
        try:
            sensors = self.sensor_manager.get_all_sensors()
            return jsonify({
                'success': True,
                'data': sensors
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def get_sensor(self, sensor_id):
        """获取指定传感器数据"""
        try:
            sensor = self.sensor_manager.get_sensor_data(sensor_id)
            
            if sensor:
                return jsonify({
                    'success': True,
                    'data': sensor
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '传感器不存在'
                }), 404
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # ========== 告警API ==========
    def get_alerts(self):
        """获取告警列表"""
        try:
            limit = request.args.get('limit', 10, type=int)
            alerts = self.alert_manager.get_recent_alerts(limit)
            
            return jsonify({
                'success': True,
                'data': alerts
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def get_alert(self, alert_id):
        """获取指定告警详情"""
        try:
            alert = self.alert_manager.get_alert(alert_id)
            
            if alert:
                return jsonify({
                    'success': True,
                    'data': alert
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '告警不存在'
                }), 404
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # ========== 历史数据API ==========
    def get_history(self, camera_id):
        """获取历史检测数据"""
        try:
            hours = request.args.get('hours', 12, type=int)
            history = self.history_manager.get_history(camera_id, hours)
            
            return jsonify({
                'success': True,
                'data': history
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def get_history_video(self, camera_id):
        """获取历史视频"""
        try:
            start_time = request.args.get('start_time')
            duration = request.args.get('duration', 60, type=int)
            
            # 从历史视频文件中提取指定时间段
            video_path = self.video_recorder.get_playback_video(
                camera_id, 
                start_time, 
                duration
            )
            
            if video_path and os.path.exists(video_path):
                return send_file(video_path, mimetype='video/mp4')
            else:
                return jsonify({
                    'success': False,
                    'error': '未找到指定时间段的视频'
                }), 404
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # ========== 系统状态API ==========
    def get_system_status(self):
        """获取系统状态"""
        try:
            # 获取当前建筑ID
            current_building_id = self.building_manager.get_current_building_id() or '未选择'
            
            # 获取摄像头数量
            total_cameras = len(self.detection_engine.cameras)
            online_cameras = len([c for c in self.detection_engine.cameras.values() if c.get('status') == 'online'])
            
            # 获取告警数量
            total_alerts = len(self.alert_manager.alerts)
            
            status = {
                'current_building': current_building_id,
                'total_cameras': total_cameras,
                'online_cameras': online_cameras,
                'offline_cameras': total_cameras - online_cameras,
                'total_alerts_today': total_alerts,
                'system_uptime': '运行中',
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify({
                'success': True,
                'data': status
            })
        except Exception as e:
            print(f"✗ 获取系统状态失败: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def stop_all_devices(self):
        """停止所有设备（摄像头、传感器）"""
        try:
            # 停止所有摄像头
            camera_ids = list(self.detection_engine.cameras.keys())
            for camera_id in camera_ids:
                self.detection_engine.stop_camera(camera_id)
            
            # 停止传感器模拟
            self.sensor_manager.stop_simulation()
            
            # 清空传感器
            self.sensor_manager.sensors.clear()
            self.sensor_manager.sensor_data.clear()
            
            return jsonify({
                'success': True,
                'message': f'已停止 {len(camera_ids)} 个摄像头和所有传感器'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def get_system_settings(self):
        """获取系统设置"""
        try:
            config = self.config_manager.get_config()
            settings = config.get('system', {
                'alert_enabled': True,
                'video_recording_enabled': False
            })
            
            return jsonify({
                'success': True,
                'data': settings
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def update_system_settings(self):
        """更新系统设置"""
        try:
            data = request.json
            
            # 获取当前配置
            config = self.config_manager.get_config()
            
            # 更新系统设置
            if 'system' not in config:
                config['system'] = {}
            
            if 'alert_enabled' in data:
                config['system']['alert_enabled'] = data['alert_enabled']
                # 更新检测引擎的告警开关
                self.detection_engine.alert_enabled = data['alert_enabled']
            
            if 'video_recording_enabled' in data:
                config['system']['video_recording_enabled'] = data['video_recording_enabled']
                # 更新视频录制器的开关
                if self.video_recorder:
                    self.video_recorder.recording_enabled = data['video_recording_enabled']
            
            # 保存配置
            self.config_manager.save_config(config)
            
            return jsonify({
                'success': True,
                'message': '系统设置已更新',
                'data': config['system']
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
