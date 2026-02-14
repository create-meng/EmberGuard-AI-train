# 数字孪生古建筑火灾监控Web系统 - 后端

## 项目结构

```
backend/
├── app.py                    # Flask主应用入口
├── config_manager.py         # 配置管理器
├── building_manager.py       # 建筑配置管理器
├── detection_engine.py       # AI检测引擎
├── alert_manager.py          # 告警管理器
├── sensor_manager.py         # 传感器管理器
├── history_manager.py        # 历史数据管理器
├── video_recorder.py         # 视频录制器
├── requirements.txt          # Python依赖
└── README.md                 # 本文件
```

## 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 运行服务器

```bash
# 开发模式
python app.py

# 生产模式（使用gunicorn）
gunicorn -k eventlet -w 1 -b 0.0.0.0:5000 app:app
```

## API端点

### 配置相关
- `GET /api/config` - 获取系统配置
- `POST /api/config` - 更新系统配置

### 建筑相关
- `GET /api/buildings` - 获取建筑列表
- `GET /api/buildings/<building_id>` - 获取建筑配置

### 摄像头相关
- `GET /api/cameras` - 获取所有摄像头状态
- `GET /api/cameras/<camera_id>` - 获取单个摄像头信息
- `GET /api/cameras/<camera_id>/status` - 获取实时状态

### 传感器相关
- `GET /api/sensors` - 获取所有传感器数据
- `GET /api/sensors/<sensor_id>` - 获取单个传感器数据

### 告警相关
- `GET /api/alerts` - 获取告警列表
- `GET /api/alerts/<alert_id>` - 获取告警详情

### 历史数据
- `GET /api/history/<camera_id>` - 获取历史检测数据
- `GET /api/history/<camera_id>/video` - 获取历史视频

### 系统状态
- `GET /api/system/status` - 获取系统状态
- `POST /api/system/demo-mode` - 切换演示模式

## WebSocket事件

### 客户端 → 服务器
- `start_video` - 启动视频流
- `stop_video` - 停止视频流

### 服务器 → 客户端
- `connected` - 连接成功
- `video_frame` - 视频帧数据
- `camera_update` - 摄像头状态更新
- `new_alert` - 新告警事件
- `sensor_update` - 传感器数据更新
- `sensor_alert` - 传感器告警

## 开发状态

当前版本：v0.1.0 - 基础架构搭建完成

- [x] Flask应用框架
- [x] 核心模块占位文件
- [ ] API路由实现
- [ ] WebSocket事件处理
- [ ] AI检测引擎集成
- [ ] 数据管理功能
